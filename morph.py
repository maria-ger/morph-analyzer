from razdel import tokenize # для разбиения текста на токены
import nltk # будет нужен Text() для составления FreqDist()
from nltk.probability import FreqDist # для подсчета числа различных слов в тексте
import string # для формирования строки из знаков препинания как стоп-слов
from pymorphy3 import MorphAnalyzer # для морфологического разбора
#from pymorphy2 import MorphAnalyzer # Для версии python 3.11 не работает MorphAnalyzer()
import matplotlib.pyplot as plt # для рисования круговых диаграмм



def word_lens(words):
    lens = [len(_) for _ in words]
    len_set = set(lens)
    len_count = dict()
    for i in range(min(len_set), max(len_set) + 1):
        len_count[i] = lens.count(i)
    return len_count

def average_word_len(all_words):
    len_count = word_lens(all_words) # словарь вида {"длина слова": "кол-во слов этой длины"}
    s = 0
    for i in len_count.keys():
        s += i * len_count[i]
    return round(s / sum(len_count.values()), 2)

def pos_of_speech_percentage(pm2, words):
    pos = [pm2.parse(word)[0].tag.POS for word in words]
    pos_set = set(pos)
    pos_percent = dict()
    for p in pos_set:
        pos_percent[p] = round(pos.count(p) / len(pos) * 100)
    return pos_percent

def feature_verb_percentage(pm2, words, feature):
    verb_feature = []
    for word in words:
        parse_res = pm2.parse(word)[0].tag
        if (parse_res.POS == 'VERB'):
            if (feature == 'tense') and (parse_res.tense != None):
                verb_feature.append(parse_res.tense)
            if (feature == 'person') and (parse_res.person != None):
                verb_feature.append(parse_res.person)
            if (feature == 'number') and (parse_res.number != None):
                verb_feature.append(parse_res.number)
    feature_set = set(verb_feature)
    feature_percent = dict()
    for f in feature_set:
        feature_percent[f] = (verb_feature.count(f) / len(verb_feature) * 100)
    return feature_percent
    
def case_percentage(pm2, words, pos):
    cases = []
    case_percent = dict()
    for word in words:
        parse_res = pm2.parse(word)[0].tag
        if parse_res.POS == pos:
            cases.append(parse_res.case)
    case_set = set(cases)
    for c in case_set:
        case_percent[c] = round(cases.count(c) / len(cases) * 100)
    return case_percent
    
    
filename = input('Программа работает с текстом из файла.\nВведите имя файла: ')

file = open(filename, 'r', encoding='UTF-8')
text = file.read().lower()
file.close()

tokens = list(tokenize(text))
alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
filtered = []
for token in tokens:
    if token.text[0] in alphabet:
        filtered.append(token.text)

text = nltk.Text(filtered)
fdist = FreqDist(text)

all_words_count = fdist.N() #встроенный метод FreqDist() - подсчет всех слов
print(f'Общее число словоупотреблений: {all_words_count}')
unique_words_count = fdist.B() #встроенный метод FreqDist() - подсчет уникальных слов
print(f'Число уникальных словоформ: {unique_words_count}')

avg_len = average_word_len(filtered) # средняя длина слов
print(f'Средняя длина слова: {avg_len}')

word_abs_freq = fdist
word_rel_freq = {key: round(fdist[key] * 100 / all_words_count, 3) for key in fdist.keys()}
n = int(input("Сколько самых частых словоформ вывести? Введите число: "))
print("Самые частые словоформы: ")
most_freq_words = sorted(word_abs_freq, key=word_rel_freq.get, reverse=True)[:min(n, unique_words_count - 1)]
# n самых частых слов
for word in most_freq_words:
    print(word, str(word_rel_freq[word]) + '%', sep=' ')

pm2 = MorphAnalyzer()

lemmas = [pm2.parse(word)[0].normal_form for word in filtered]
lemmas = nltk.Text(lemmas)
fdist_lemm = FreqDist(lemmas)

unique_lemmas_count = fdist_lemm.B() #подсчет уникальных лемм
print(f'Число уникальных лемм: {unique_lemmas_count}')
# абсолютная и относительная частоты лемм относительно словоупотреблений (в процентах)
lemma_abs_freq = fdist
lemma_rel_freq = {key: round(fdist_lemm[key] * 100 / all_words_count, 3) for key in fdist_lemm.keys()}
n = int(input("Сколько самых частых лемм вывести? Введите число: "))
print("Самые частые леммы: ")
most_freq_lemmas = sorted(lemma_rel_freq, key=lemma_rel_freq.get, reverse=True)[:min(n, unique_lemmas_count - 1)] 
# n самых частых лемм
for lemma in most_freq_lemmas:
    print(lemma, str(lemma_rel_freq[lemma]) + '%', sep=' ')

pos_percent = pos_of_speech_percentage(pm2, filtered) # dict {"часть речи": "процент"}

noun_case_percent = case_percentage(pm2, filtered, 'NOUN')
most_freq_noun_case = max(noun_case_percent)
print(f'Самый частый падеж у имен сушествительных: {most_freq_noun_case}')

adjf_case_percent = case_percentage(pm2, filtered, 'ADJF')
most_freq_adjf_case = max(adjf_case_percent)
print(f'Самый частый падеж у имен прилагательных: {most_freq_adjf_case}')

tense_percent = feature_verb_percentage(pm2, filtered, 'tense')
most_freq_tense = max(tense_percent)
print(f'Самое частое время у глаголов: {most_freq_tense}')

person_percent = feature_verb_percentage(pm2, filtered, 'person')
most_freq_person = 0
if len(person_percent):
    most_freq_person = max(person_percent)
    print(f'Самое частое лицо у глаголов: {most_freq_person}')

number_percent = feature_verb_percentage(pm2, filtered, 'number')
most_freq_number = max(number_percent)
print(f'Самое частое число у глаголов: {most_freq_number}')

plt.figure(figsize=(20, 10))

colors = ['r', 'lightgreen', 'y', 'purple', 'pink', 'coral', 'cyan', 'gray', 'teal', 'b', 'saddlebrown', 'rosybrown', 'khaki', 'deeppink', 'darkgreen', 'cornflowerblue', 'maroon']
pos_percent = dict(sorted(pos_percent.items(), key=lambda item: item[1], reverse=True))
pos = [str(p) for p in pos_percent]
sizes = list(pos_percent.values())
labels = [f"{p} {s}%" for p,s in zip(pos, sizes)]
ax = plt.subplot(2, 3, 1)
plt.title('Части речи')
ax.pie(sizes, colors=colors)
plt.legend(labels=labels, bbox_to_anchor=(0, 1))

noun_case_percent = dict(sorted(noun_case_percent.items(), key=lambda item: item[1], reverse=True))
labels = [str(p) for p in noun_case_percent]
sizes = list(noun_case_percent.values())
ax = plt.subplot(2, 3, 2)
plt.title('Падежи существительных')
ax.pie(sizes, labels=labels, autopct='%.0f%%')

labels = [str(p) for p in adjf_case_percent]
sizes = list(adjf_case_percent.values())
ax = plt.subplot(2, 3, 3)
plt.title('Падежи прилагательных')
ax.pie(sizes, labels=labels, autopct='%.0f%%')

tense_percent = dict(sorted(tense_percent.items(), key=lambda item: item[1], reverse=True))
labels = [str(p) for p in tense_percent]
sizes = list(tense_percent.values())
ax = plt.subplot(2, 3, 4)
plt.title('Времена глаголов')
ax.pie(sizes, labels=labels, autopct='%.0f%%')

labels = [str(p) for p in person_percent]
sizes = list(person_percent.values())
ax = plt.subplot(2, 3, 5)
plt.title('Лица глаголов')
ax.pie(sizes, labels=labels, autopct='%.0f%%')

labels = [str(p) for p in number_percent]
sizes = list(number_percent.values())
ax = plt.subplot(2, 3, 6)
plt.title('Числа глаголов')
ax.pie(sizes, labels=labels, autopct='%.0f%%')

plt.show()

answer = input("Хотите определить стиль текста? Введите yes или no.\n")
if answer == 'yes':
    noun_case_percent_key_list = list(map(lambda x: x[0], noun_case_percent.items()))
    pos_percent_key_list = list(map(lambda x: x[0], pos_percent.items()))
    tense_percent_key_list = list(map(lambda x: x[0], tense_percent.items()))
    if ('VERB' in pos_percent) and (pos_percent['VERB'] >= 10) and ((pos_percent_key_list[1] in ['ADJF', 'ADJS']) or (pos_percent_key_list[2] in ['ADJF', 'ADJS'])):
        print('Художественный')
    elif (noun_case_percent_key_list[0] == 'gent') or (('1per' in person_percent) and (person_percent['1per'] < 5)) or (('2per' in person_percent) and (person_percent['2per'] < 5)) or (tense_percent_key_list[0] == 'pres'):
        print('Научный')
    else:
        print("Не удалось определить.\nПрограмма завершена.")
else:
    print("Программа завершена.")