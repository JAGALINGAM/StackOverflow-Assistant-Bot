import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

from utils import *
from sklearn.metrics.pairwise import cosine_similarity


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embedding("starspace_embedding.tsv")
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        
        question_vec = np.array(question_to_vec(question=question, embeddings=self.word_embeddings, dim=self.embeddings_dim))
        #print(question_vec.shape)
        #print(thread_embeddings.shape)
        question_vec = question_vec.reshape(-1,100)
        siml = pairwise_distances_argmin(question_vec,thread_embeddings)
        best_thread = siml[0]
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.tfidf_vectorizer = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER'])
        self.intent_recognizer = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(RESOURCE_PATH)
        self.create_chitchat_bot()

    def create_chitchat_bot(self):

        self.chitchat_bot = ChatBot("Kenpachi Zaraki")
        self.trainer = ChatterBotCorpusTrainer(self.chitchat_bot)
        self.trainer.train("chatterbot.corpus.english")
		
		
       
    def generate_answer(self, question):
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':     
            response = self.chitchat_bot.get_response(question)
            return response
        
        else:        
            tag = self.tag_classifier.predict(features)[0]
            
            thread_id = self.thread_ranker.get_best_thread(prepared_question,tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)
        
'''dialaogue_manager = DialogueManager(RESOURCE_PATH)
question = input()
print(question)
answer = dialaogue_manager.generate_answer(question)
print(answer)

path = os.path.join("thread_embeddings_by_tags","c#"+".pkl")
thread_ids, thread_embeddings = unpickle_file(path)'''