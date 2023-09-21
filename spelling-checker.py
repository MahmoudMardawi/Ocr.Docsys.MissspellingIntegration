import tensorflow
print(tensorflow.__version__)    # >=2.3.1

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
from flask import Flask, request
import random
os.environ["PYTHONIOENCODING"] = "utf-8"
import string
import nltk
from nltk.stem.isri import ISRIStemmer
st = ISRIStemmer()

#### Do this once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading the spelling the checker model...")
model = load_model("model")


print("Building the tokenizer...")
data_file = "data/News-Multi.ar-en.ar.more.clean"
data = open(data_file,encoding="utf8").read()

corpus = data.lower().split("\n")
#print("First sentence in the corpus:", corpus[0])

vocab_size = 100000 
max_sequence_len = 15
out_of_vocab = "<unk>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=out_of_vocab)
tokenizer.fit_on_texts(corpus)


# RWE
texts_to_correct = [ "لام مجلس الوژراء",
 "۹ا ع.",
" ل بالمرسوم الملكي" ,
" بات الاستم ",
 "کين بغراو ف" ,
" ويعد الاطلاع على الفقرة (۲) من المادة ‎)۲٤(‏ من ذ" ,
" الصأدر بالامر الملكي رقم ۳/۶ \( وتاريخ ‎٤١٤/٤٣/٣۳‏ ١ه‏ ",
 "ويعد الاطلاع على الامر الملكي رقم 6۲/8 ‎)٤‏ وتاريخ ‎٠۱۹‏ ",
 "ويعد الاطلاع على نظام التأمينات الاجتماعية؛ الصا" ,
 "رقم 2م۳۳( وتاريخغ ‎٠٤١ ٧/۹/۳‏ ه؛ وتعديلاته." ,
 "ويعد الاطلاع على المرسوم الملكي رقم )م /£ 9( وتاريخ" ,
 "المعدل بالمرسوم الملكي رقم (م/4 0( وتاريخ ‎٤١ ٤/۸/٠١‏ ٠ه.‏" ,
 "ويعد الاطلاع على ترار مجلس الوزراء رقم ‎)۱۹١(‏ وتاريخ" ,
 "ويعد الطلاع على المذكرة رقم ‎)1۱٠١(‏ وتاريخ ‎۲/٤/١١" ,
" هيثة الخبراء بمجلس الوزراء." ,
 "ويعد الأطلاع على التوصية المعدة في مجلس الشؤون" ,
 "رقم ‎٦(‏ 4/۸ /د) وتاريځخ ‎٠٤٤۲/٥/۲‏ ه. ",
 "وبعد الاطلاع على توصية اللجشة العامة لمجلس ال" ,
 "تعديل الفقرتين (۱) و(۲) من المادة (العاشرة) من نظام 1",
 "¦ -الصادر بالمرسوم الملكي رقم (م/۳۳) وتاريخ 1/۹/۳ "  ,
 "الوزراء رقم ‎)٠۹(‏ وتاریځ ‎٤٣۳۸/۳/۲٢‏ ١ه‏ لتكونا بالتص أا تى"
                    ]

# NWE
texts_to_correct = [ "لام مجلس الوژراء",
 "۹ا ع.",
" ل بالمرسوم الملكي" ,
" بات الاستم ",
 "کين بغراو ف" ,
" ويعد الاطلاع على الفقرة (۲) من المادة ‎)۲٤(‏ من ذ" ,
" الصأدر بالامر الملكي رقم ۳/۶ \( وتاريخ ‎٤١٤/٤٣/٣۳‏ ١ه‏ ",
 "ويعد الاطلاع على الامر الملكي رقم 6۲/8 ‎)٤‏ وتاريخ ‎٠۱۹‏ ",
 "ويعد الاطلاع على نظام التأمينات الاجتماعية؛ الصا" ,
 "رقم 2م۳۳( وتاريخغ ‎٠٤١ ٧/۹/۳‏ ه؛ وتعديلاته." ,
 "ويعد الاطلاع على المرسوم الملكي رقم )م /£ 9( وتاريخ" ,
 "المعدل بالمرسوم الملكي رقم (م/4 0( وتاريخ ‎٤١ ٤/۸/٠١‏ ٠ه.‏" ,
 "ويعد الاطلاع على ترار مجلس الوزراء رقم ‎)۱۹١(‏ وتاريخ" ,
 "ويعد الطلاع على المذكرة رقم ‎)1۱٠١(‏ وتاريخ ‎۲/٤/١١" ,
" هيثة الخبراء بمجلس الوزراء." ,
 "ويعد الأطلاع على التوصية المعدة في مجلس الشؤون" ,
 "رقم ‎٦(‏ 4/۸ /د) وتاريځخ ‎٠٤٤۲/٥/۲‏ ه. ",
 "وبعد الاطلاع على توصية اللجشة العامة لمجلس ال" ,
 "تعديل الفقرتين (۱) و(۲) من المادة (العاشرة) من نظام 1",
 "¦ -الصادر بالمرسوم الملكي رقم (م/۳۳) وتاريخ 1/۹/۳ "  ,
 "الوزراء رقم ‎)٠۹(‏ وتاریځ ‎٤٣۳۸/۳/۲٢‏ ١ه‏ لتكونا بالتص أا تى"

                    ]



texts_to_correct =  ["المدفوعات رتم ‎١‏ وباريح ‎٠/٠١‏ 8 اها فى سان المتطليات الماسيسية د مانة لجنة |", 
 "التوطين ومهزان المدفوعات.", 
 "وبعد الاطلاع على المذكرات رقم (۷) وتاری ‎٤٤/١١/۷‏ اه ورلم )۳۲۳(", 
 "وتاریځ ‎٤٤٦٤/۲/۲۷‏ ١ےا‏ ورقم (1) وتساريځ ‎٤٤/٤/١٤‏ اه؛ المصدة في",
 "ويمد الأطلاع على المحضسر المعد في مجلس الشؤون الاقتصادية والتتمية |", 
 "رقم ‎٤۲/٥ ٥(‏ لم) وتاريخ ‎٠٤٤٦/۳/٢‏ ه. ",
 "ويعد الأطظلاع على توسصية اللجنة العامة لمجلس الوزراء رقم ‎٤٢ ٤(‏ ۲) "
 "وتاريځ ‎٠٤٤١/٤/۱۷‏ ه. ",
 "يقرر ما يلي:" 
 "أولاً : إنشاء مكتب باسم (مكتب التوطين وميزان المدفوعات)› يكون مرتبطاً بلجنة ",
 "التوطين وميىزان المدفوعات» ويكون له تنظيم إداري ومالي مستقل وميزانية ",
 "مستقلةء ويتولى إدارة أعمال أمائة اللجنة؛ ويكون معالي آمين عام لجنة التصوطين", 
 "وميزان المدفوعات مشرفا عاما عليه.", 
 "ثائيا : منح معالي أمين عام لجثة التوطين وميزان المدفوعات الصلاحيات الأتية:", 
 "‎١‏ - الاستقطاب وطلب الندب والإعارة والتعاقد مع:", 
 "] - الخبراء والمختصين والمستشارين المحليين بمن فيهم: العاملون في", 
 "الجنهات الحكومية أو أعضاء هيثة التدريمن أو غیرهم." ]


texts_to_correct = ["کی س اک ی ل",
 "معالي / نائب وزير الخارجية رئيساً للجنة الوطدية لمتابعة مبادرة خادم الحرمين الشريغين",
 "الملك عبدالك بن عبدالعزيز آل سعود للحولار بين أتباع الأديان والعقانات› وضم ممٹل من", 
 "وزارة الخارجية إلى عضوية اللسنة المشار إليها", 
 "وبعد الاطلاع على تنظيم اللجنة الوطنية لمتايعة ميادرة خادم الحرمين الشريقين",
"| الملك عبدالله ين عبدالعزيز آل سعود للحوار بين أتباع الأديان والثقانات؛› الصادر بقرار",
 "مجلس الوزراء رقم .69 وتاريخ ۷/۳ اه",
 "ويعد الاطلاع على برقية أمانة مجلس الشؤون السياسية والامنية رقم ۹۲۷",
 "وتاریخ ‎٠٤٤٤/۳/۲۸‏ ه.",
 "ويعد الاطلاع على توصة اللجنة العامة لمجلىس الوزدراء رقم ‎(۲٤٢ ٤٤٥٤(‏",
"وتاريخ ‎٠٤٤٦/٤/٧١‏ ه."
 "يعرر",
 "تعديل الفقرتين ‎)١(‏ و(۲) من المادة (الثالثة) من تنظيم اللجنة الوطنية لمتايعة ميادرة",
 "خادم الحرمين الشريفين الملك عبدالله بن عبدالعزيز آل سعود للحوار بين أتباع الأديان", 
" والثقافات -الصادر بقرار مجلس الوزراء رقم ‎)٠١(‏ وتاريخ ‎٤۳۷/۱١/١١‏ ٠ه‏ لتكونا",
 "| بالنص الآتي: ",
 "| -تشكل اللجنة الوطنية على الحو الاقي: ۱ ",
 "أ تاڻپ وزير الخارجية رئيسا",
 "ب -الآأمين العام لمركز الملك عبدالله بن عبدالعزيز" ,
 "العالمي للحوار بين أتباع الاديان والثقانات عضوا. ",
 "‎J‏ ح -الامين العام لمركز اله الملك عبدالعزيز للحوار الوطني عضوا."] 
fileEngine = str(random.randint(0,1000))+'.txt'

RFileText = str;

app = Flask(__name__)
# create folder for uploaded data
FOLDER = 'uploaded'
os.makedirs(FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/fileupload', methods=['POST'])
def index():
        #strBody = str   
        with open("test.txt", "w",encoding="utf8") as text_file:
                  #test = text_file.write(request.files.get('file').read().decode());
                  file = request.files.get('file')
                  lines = file.readlines()
                  OceEngine(lines)
                  
                  #with open("DocsysBackup(1).txt", "r",encoding="utf8") as txt_file_1:
                      #body = file.readlines()
                      #txt_file.close()
                      #print(body)
                  with open("DocsysBackup.txt", 'rb') as file_t:
                            blob_data = bytearray(file_t.read())
                            return blob_data
                  
        return "NotFound"
def generate_ngrams(text):
    text = "<s>" + text
    
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    all_grams = []

    for n in range(2, len(tokens)+1):
        grams = [w for w in nltk.ngrams(tokens, n=n)][0]
        
        grams_rev = tokens[n:]
        grams_rev.reverse()
        all_grams.append((list(grams), list(grams_rev)))

    return all_grams

def OceEngine(text):
          with open("DocsysBackup.txt", "w+",encoding="utf8") as text_file:            
                for text_to_correct in text:
    
                    words = word_tokenize(text_to_correct.decode('utf-8'))

                    text_to_correct = ' '.join([word for word in words if len(word) > 1])
                    text_file.write('\n')
                    print("Currently correcting:", text_to_correct)
                    text_file.write(" Currently correcting : "+text_to_correct)
                    text_file.write('\n')
                    ngrams = generate_ngrams(text_to_correct)
                    correct = None
                    suggestions = []
    
                    for ngram in ngrams:
            
                        if len(ngram[0]) > 2 and correct != 1 and len(suggestions) != 0:
                            seed_text_ltr = " ".join(word for word in ngram[0][:-2]) + " " + suggestions[0][2]
                        else:
                            seed_text_ltr = " ".join(word for word in ngram[0][:-1])
        
                        current_word = ngram[0][-1]
                        seed_text_rtl = " ".join(word for word in ngram[1])
                        text_file.write('\n')
                        print(seed_text_ltr, "->", current_word, "->", seed_text_rtl)
                        text_file.write(seed_text_ltr +  " -> "+ current_word +  " -> " + seed_text_rtl)
                        text_file.write('\n')
                        token_list = tokenizer.texts_to_sequences([seed_text_ltr])[0]
                        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

                        token_list_rev = tokenizer.texts_to_sequences([seed_text_rtl])[0]
                        token_list_rev = pad_sequences([token_list_rev], maxlen=max_sequence_len-1, padding='pre')

                        predicted_id = np.argmax(model.predict([token_list, token_list_rev]), axis=-1)
                        predicted_word = tokenizer.sequences_to_texts([predicted_id])[0]
                        print(predicted_word)


                        predicted_probs = model.predict([token_list, token_list_rev])
                        predicted_best = np.argsort(-predicted_probs, axis=-1)[0][:4500]
        
                        suggestions = []
                        correct = None

                        for prob in predicted_best:
                            output_word = tokenizer.sequences_to_texts([[prob]])[0]
                            ed = nltk.edit_distance(current_word, output_word)

                            if ed ==0:
                                #print("I got this one; it seems correct -->", current_word, "=", output_word)
                                with open(fileEngine, 'w',encoding="utf8") as f:
                                      text_file.write('\n')  
                                      f.writelines(["Current:"+current_word,  "Output:"+output_word])
                                      text_file.write("Current:  "+current_word +   "   Output:  "+output_word)
                                      text_file.write('\n')
                                      #f.close()
                                correct = 1
                                break
                            elif len(current_word)<=3 and ed ==1:
                                suggestions.append((ed, current_word, output_word))
                            elif len(current_word)>3 and ed <=2:
                                suggestions.append((ed, current_word, output_word))
                            else:
                                continue
                
        
        
                        if len(suggestions) > 0:  
                            for suggest in suggestions:
                                lemmas_cw = []
                                lemmas_cw.append(suggest[1])
                                lemmas_cw.append(st.suf1(suggest[1]))
                                lemmas_cw.append(st.suf32(suggest[1]))
                                lemmas_cw.append(st.pre1(suggest[1]))
                                lemmas_cw.append(st.pre32(suggest[1]))
                
                                lemmas_ow = []
                                lemmas_ow.append(suggest[2])
                                lemmas_ow.append(st.suf1(suggest[2]))
                                lemmas_ow.append(st.suf32(suggest[2]))
                                lemmas_ow.append(st.pre1(suggest[2]))
                                lemmas_ow.append(st.pre32(suggest[2]))
                
                                if correct != 1 and len(suggest[1]) > 7:
                                    for l in lemmas_cw:
                                        if l in lemmas_ow:
                                            correct = 2
                                            text_file.write('\n')
                                            print("I got the lemma; it seems correct -->", current_word, "~", suggest[2])
                                            text_file.write(" I got the lemma; it seems correct -->  " +  current_word +  "  ~   " +  suggest[2])
                                            text_file.write('\n')

                        print("Suggestions:", " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1]))
        
                        #RFile = open(fileEngine,"w+",encoding="utf8")
                        L = " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1])
                        text_file.write('\n')
                        text_file.write("Suggestions:" +  " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1]))
                        text_file.write('\n')
                        #text_file.close()
        
                        if correct == 2:
                            print("Not sure")
                            text_file.write('\n')
                            #RFile = open(fileEngine,"w+",encoding="utf8")
                            L = " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1])
                            text_file.write("Not sure")
                            text_file.write('\n')
                            #text_file.close()
                        elif correct == 1:
                            
                            print("CORRECT")
                            #RFile = open(fileEngine,"w+",encoding="utf8")
                            text_file.write('\n')
                            L = " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1])
                            text_file.write("CORRECT")
                            text_file.write('\n')
                            #text_file.close()
                        elif correct != 1 and len(suggestions) > 0:
                            correct = 0
                            #RFile = open(fileEngine,"w+",encoding="utf8")
                            L = " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1])
                            text_file.write('\n')
                            text_file.write("WRONG")
                            text_file.write('\n')
                            #text_file.close()
                            print("WRONG")
                        elif correct != 1 and len(suggestions) == 0:
                            print("I do not know!")
                            #RFile = open(fileEngine,"w+",encoding="utf8")
                            text_file.write('\n')
                            L = " - ".join([suggest[2] for suggest in suggestions if len(suggest[2]) > 1])
                            text_file.write("I do not know!")
                            text_file.write('\n')
                            #text_file.close()
                     
                        
                        print("-------")
                        
                #text_file.close()  
                #
                text_file.close()

app.run(port=8090)
