# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
import random
import numpy as np
import re
from nltk.stem.porter import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Z-bot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.seen_movies = [0] * 9125
        self.processing_title = ""
        self.rec_index = 0

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_messages = [
            "Yo, Z-bot in the house! Whatcha need?",
            "Sup, it's Z-bot here. Hit me up with your queries!",
            "Heyo, Z-bot at your service. What's poppin'?",
            "Wassup, it's ya boi Z-bot. How can I slide into your DMs today?",
            "Ayy, Z-bot's the name, helping's my game. What's good?",
            "Yo yo yo, Z-bot's here to glow. What you wanna know?",
            "Hey there, Z-bot's on the beat. What's the word on the street?",
            "What's crackalackin'? Z-bot's here to chat. What's on your mind?",
            "Aloha, Z-bot's in the vibe. What's the sitch?",
            "Howdy, Z-bot's in the loop. What's brewing?"
            ]

        random_greeting = random.choice(greeting_messages)
        return random_greeting

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Cheers! See you soon."

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################


    @staticmethod
    def get_feeling(sentiment):
        dict_feelings = {
            -1: ["you didn't vibe with", " you were meh about", "you lowkey hated"],
            1: ["were totally vibing with", "were stoked about", "were low-key obsessed", "were feeling", "were here for", "were hella into", "couldn't get enough of", "ate up"]
        }
        random_word = random.choice(dict_feelings[sentiment])
        return random_word

    @staticmethod
    def prompt_next():
        words = ["Tell me about another movie.", "Spill the tea on another flick you've checked out.", "Spill the beans on another movie you've seen.", "Mind giving me another movie?"]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def movie_notfound ():
        words = ["Sorry dude, not familiar with that movie.", "Sorry, dude. Not familiar with that movie.",
            "My bad, homie. Never heard of it.",
            "Oops, I'm out of the loop on that one.",
            "Apologies, Never came across it.",
            "My apologies, but that title doesn't sound familiar.",
            "Ah, that's a new one for me. Haven't seen it.",
            "Sorry, mate. Not in my movie repertoire.",
            "My bad, but I'm clueless about that movie.",
            "Sorry, I'm not up to speed on that one.",
            "Ah, that's a mystery to me. Never seen it.",
            "Sorry, but I'm not in the know about that film.",
            "Ah, my bad. Not part of my movie knowledge.",
            "Sorry, I'm drawing a blank on that movie.",
            "My apologies, but I'm not familiar with that flick."
            ]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def neutral_sentiment(title):
        words = [
            "Hey,\" {} \" was like, okay I guess.".format(title),
            "Um, \"{}\" was kinda meh.".format(title),
            "\"{}\" it's whatever.".format(title),
            "\"{}\" was chill, but not great.".format(title),
            "I guess \"{}\" was alright.".format(title),
            "\"{}\" was like, fine.".format(title),
            "Yeah, \"{}\" was pretty average.".format(title),
            "Bruh, \"{}\" is, like, okay.".format(title),
            "\"{}\" just kills some time.".format(title),
            "\"{}\" was kinda mid.".format(title),
            "\"{}\" was, like, decent.".format(title),
            "\"{}\" was like chill for a background watch.".format(title)
        ]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def sentiment_not_understood(title):
        """Returns a message for not understanding the user's movie preference."""
        words = [
            "Bruh, my bad. Not sure if you liked \"{}\". Wanna tell me more about it?".format(title),
            "Oof, sorry. Can't tell if you vibe with \"{}\". Can you tell me more?".format(title),
            "Hold up, so did you like \"{}\" or nah? Tell me more.".format(title)
        ]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def not_movie():
        words = ["Dawg... I don't see a movie in there.",
                 "Hmm, seems like we're missing a movie.", 
                 "Whoops! Looks like we missed the movie title.", 
                 "Wait a sec, did we forget the movie?",
                 "Hey, where did the movie title go?",
                 "Oops, did we forget to mention the movie?",
                 "Hmm, no movie in sight.", 
                 "Hold up, no movie in the mix.", 
                 "Seems like there's a missing piece here... oh right, the movie! What's the title?",
                 "Hey, I'm missing the movie in this conversation. What's the title?", 
                 "My bad, didn't catch the movie title. Mind sharing it again?",
                 "Looks like the movie got lost in translation. Can you remind me of the title?",
                 "Hmm, no movie mentioned!"]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def too_many_movies():
        words = ["Hold up, can we take this one movie at a time?", 
                 "Wait a sec, can we slow down and go one by one?",
                 "Yo, can we chill and go one at a time? ", 
                 "Hey, can we do this sequentially, like one by one? ", 
                 "Pause, can we go through this one movie at a time? ", 
                 "Hang on, let's go through this in order, one at a time.", 
                 "Hold on, can we go through this sequentially, movie by movie?",
                 "Hey, can we take it easy and go through this one movie at a time?", 
                 "Wait a minute, can we go through this one movie at a time, please?"]
        random_word = random.choice(words)
        return random_word

    @staticmethod
    def ambiguous_movies(title_matrix, movies):
        response = ""
        response_options = ["I don't know if you're taking about: ", 
                            "Are you talking about: ", 
                            "There are several movies you could be referring to. Are you talking about: "] 
        response += random.choice(response_options)
        for i in range(len(movies) - 1):
            index = movies[i]
            response += title_matrix[index][0] + ","
        response += "or " + str(title_matrix[len(movies) - 1][0]) + "?"
        return response

    @staticmethod
    def get_recommendations(recs, titles, index):
        """Giving recommendations (asks automatically after user provides 5 data points) The bot should give one recommendation at a time, each time giving the user a chance to say "yes" to hear another or "no" to finish."""
        # Assuming `ratings_matrix` is accessible within this instance method and `rec_index` is meant to be `index`.
        recommend_statements = ["I think you would be obsessed with", "You would vibe with", "You would ADORE"]
        another_statements = ["Would you like to hear another rec?", "Want another movie rec?", "Should I give you one more recommendation?"]

        # Ensure `index` is within the bounds of `recs`.
        if index < len(recs):
            response = random.choice(recommend_statements) + " " + titles[recs[index]][0] + "? " + random.choice(another_statements) + " Answer Yes or No."
        else:
            response = "I've run out of recommendations for now. Let's talk about something else!"
        return response
    
    @staticmethod
    def get_acknowledgement():
        words = ["Rad", "Dope", "Fire", "Yaaaas", "Ok cool", "My g", "Beaut", "Tecky", "Ok, facts"
            ]
        random_word = random.choice(words)
        response = random_word + ", so you "
        return response


    @staticmethod
    def goodbye():
        words = ["Aight, see ya!", "Kk, good talk.", "Okay, byeeee."
            ]
        random_word = random.choice(words)
        response = random_word + " Hope to be able to help you in the future"
        return response
    
    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
            1) extract the relevant information, and
            2) transform the information into a response to the user.

        Example:
            resp = chatbot.process('I loved "The Notebook" so much!!')
            print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            response = "I processed {} in Starter (GUS) mode!!".format(line)

        if sum(1 for mov in self.seen_movies if mov != 0) == 5 and line[0].lower() == 'y':
            self.rec_index += 1
            recs = self.recommend(self.seen_movies, self.ratings, k=100, llm_enabled=False)
            return self.get_recommendations(recs, self.titles, self.rec_index)

        elif sum(1 for mov in self.seen_movies if mov != 0) == 5 and line[0].lower() == 'n':
            self.rec_index = 0
            self.seen_movies = []
            return self.goodbye() + "Type in \":quit\" to exit the chat session."
    
        elif sum(1 for mov in self.seen_movies if mov != 0) == 5:
            recs = self.recommend(self.seen_movies, self.ratings, k=100, llm_enabled=False)
            self.get_recommendations(recs, self.titles, self.rec_index)
            return self.get_recommendations(recs, self.titles, self.rec_index)


        # construct response from user input
        title = self.extract_titles(line) 
        sentiment = self.extract_sentiment(line)

        ## TITLE
        # Title options
        # "i like pink"
        if (self.processing_title == ""):
            if title == []:
                return self.not_movie() + " " + self.prompt_next()

            # More than one title given
            elif len(title) > 1:
            #handle_multiple_titles
                return self.too_many_movies() + " " + self.prompt_next()
            
            str_title = title[0]
            # Movie title is ambiguos i.e. Titanic but there are several releases of the movie
            if (len(self.find_movies_by_title(str_title)) > 1):
                return self.ambiguous_movies(self.titles, self.find_movies_by_title(title))
                
            # Movie not in database
            elif self.find_movies_by_title(str_title) == []:
                return self.movie_notfound() + " " + self.prompt_next()
            else:
                self.processing_title = str_title

        ##SENTIMENT
        #Sentiment options
        if (self.processing_title != ""):
            if sentiment == 0:
                return self.sentiment_not_understood(self.processing_title) #self.neutral_sentiment(title) + self.sentiment_not_understood(title)
            # fill sentence, acknoweledge
            else: 
                movie_index = (self.find_movies_by_title(self.processing_title))[0]

            #self.movies_index.append(movie_index)
            self.seen_movies[movie_index] = sentiment
            response = self.get_acknowledgement() + " " + self.get_feeling(sentiment) + " \"" + self.processing_title + ".\" " + self.prompt_next()
            self.processing_title = ""
            return response
        return "I didn't quite understand, can you try inputting something different?"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        #  text = text.split()
        return text

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
            potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
            print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = []
        regex = "\"(.*?)\""
        titles = re.findall(regex, preprocessed_input)

        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
            ids = chatbot.find_movies_by_title('Titanic')
            print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        title_words, *year = title.split(' (', 1) 
        year = '(' + ''.join(year) if year else ""

        if title.find('(') != -1 :
            year_start = title.find('(')
            title_words = title[:year_start - 1] #exclude space
            year = title[year_start:]
        
        if title_words.split()[0] == "A" or title_words.split()[0] == "An" or title_words.split()[0] == "The":
                title_words = ' '.join(title_words.split()[1:]) + ", " + title_words.split()[0]

        if year != "":
            title_words += " " + year
            return [i for i in range(len(self.titles)) if self.titles[i][0] == title_words]
        else: 
            return [i for i in range(len(self.titles)) if re.search(title_words + "\s\(\d{4}\)", self.titles[i][0]) != None
                    or self.titles[i][0] == title_words]

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
            sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
            print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        score = 0
        negated = False
        negations = ["not", "isn't", "wasn't", "don't", "doesn't", "didn't", "haven't", "hasn't", "never"]
        sentiment_map = {"pos": 1, "neg": -1}
        updated_input = re.sub(r'\"[^\"]+\"', ' ', preprocessed_input)
        array_words = updated_input.split()
        stemmer = PorterStemmer()

        for word in array_words:
            word = word.lower()
            word = stemmer.stem(word)
            if word in negations:
                negated = True
            elif word in self.sentiment:
                sentiment = self.sentiment[word]
                sentiment_value = sentiment_map[sentiment]
                if negated:
                    sentiment_value *= -1
                    negated = False
                score += sentiment_value
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
            0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings = np.where(ratings > threshold, 1, binarized_ratings)
        binarized_ratings = np.where((ratings <= threshold) & (ratings != 0), -1, binarized_ratings)
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        similarity = np.dot(u, v) / (norm_u * norm_v)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
            filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
            `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        recommendations = []
        already_rated = np.where(user_ratings != 0)[0]

        for i in range(len(user_ratings)):
            if i not in already_rated:
                similarity_sum = 0
                for j in already_rated:
                    if np.linalg.norm(ratings_matrix[i]) != 0 and np.linalg.norm(ratings_matrix[j]) != 0:
                        similarity_sum += self.similarity(ratings_matrix[i], ratings_matrix[j]) * user_ratings[j]
                recommendations.append((i, similarity_sum))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:k]

        return [movie[0] for movie in recommendations]

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
        'run:')
    print('    python3 repl.py')