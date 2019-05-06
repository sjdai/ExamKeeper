

from docx import Document

document = Document(u'test.docx')

voc = False
mul = False
cloze = False
passage = False
read = False
Fit = False
Translation = False

for para in document.paragraphs:
    para = para.text
    temp = str(para).lower()
    if para = ' ' or '':
        continue
    elif 'vocabulary' in para:
        voc = True
        mul = False
        cloze = False
        passage = False
        read = False
        Fit = False
        Translation = False
    elif 'multiple choice' in para:
        voc = False
        mul = True
        cloze = False
        passage = False
        read = False
        Fit = False
        Translation = False
    elif 'passage completion' in para:
        voc = False 
        mul = False
        cloze = False
        passage = True
        read = False
        Fit = False
        Translation = False
    elif 'cloze' in para:
        voc = False
        mul = False
        cloze = True
        passage = False
        read = False
        Fit = False
        Translation = False
    elif 'reading comprehension' in para:
        voc = False
        mul = False
        cloze = False
        passage = False
        read = True
        Fit = False
        Translation = False
    elif 'fit the best sentence' in para:
        voc = False
        mul = False
        cloze = False
        passage = False
        read = False
        Fit = True
        Translation = False
    elif 'translation' in para:
        voc = False
        mul = False
        cloze = False
        passage = False
        read = False
        Fit = False
        Translation = True
    if voc:

    elif mul:
    elif cloze:
    elif passage:
    elif read:
    elif Fit:
    elif Translation:










