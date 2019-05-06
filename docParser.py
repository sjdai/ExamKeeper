import docx

questionType = ['vocabulary','multiplechoice','cloze','passagecompletion','readingcomprehension','fitthebestsentence','translation']
optionType = ['(A)','(B)','(C)','(D)','(E)','(F)','(G)','(H)','(I)','(J)','(K)','(L)','(M)','(N)','(O)','(P)','(Q)','(R)','(S)','(T)','(U)','(V)','(W)','(X)','(Y)','(Z)']

def retrieveIndex(context):
    idx = []
    if 'are based on the following passage.' in context:
        context = context.replace('-',' ')
        context = context.split()
        idx = [i for i in context if i.isdigit()]
    return idx

def isOption(context):
    val = False
    for o in optionType:
        if o in context:
            val = True
    return val

def indexer(context,point):
    paragraphs = [i.replace(' ','') for i in context]
    index = [[i,paragraphs.index(i)]for i in point if i in paragraphs]
    return index

def ParseVoc(context,name):
    context = [i for i in context if i != '']
    vocAnswer = ans['Vocabulary']
    questions = []
    context.pop(0)

    for term in range(len(context)):
        tmp = context[term]
        tmp = tmp.split()
        for t in tmp:
            if '_' in t:
                tmp[tmp.index(t)] = vocAnswer[term]
        questions.append({'sentence':' '.join(tmp),'answer':vocAnswer[term]})

    return questions

def ParseMultiple(context,name):
    context = [i for i in context if i != '']
    Answer = ans[name]
    answerId = 0
    questions = []
    context.pop(0)
    article = ''
    for term in range(len(context)):
        tmp = context[term]
        tmp = tmp.split()
        key = optionType[0] in context[term]
        if key:
            index = indexer(tmp,optionType)
            for i in index:
                tmp[i[1]] = ''
            tmp = [i for i in tmp if i != '']
            tmp.pop(0)
            questions.append({'article':article,'answer':Answer[answerId],'option':tmp})
            answerId += 1
        else:
            for t in tmp:
                if '_' in t:
                    tmp[tmp.index(t)] = '_'
                    article = ' '.join(tmp)
    return questions

def ParseCloze(context,name):
    context = [i for i in context if i != '']
    Answer = ans[name]
    answerId = 0
    questions = []
    context.pop(0)
    article = ''
    context.pop(0)
    option = []
    answer = []
    appendFlag = False
    for term in range(len(context)):
        tmp = context[term]
        tmp = tmp.split()
        key = optionType[0] in context[term]
        if key:
            appendFlag = True
            index = indexer(tmp,optionType)
            for i in index:
                tmp[i[1]] = ''
            tmp = [i for i in tmp if i != '']
            tmp.pop(0)
            option.append(tmp)
        else:
            if appendFlag:
                appendFlag = False
                questions.append({'article':article,'answer':answer,'option':option})
                article = ''
                option = []
                answer = []
            for t in tmp:
                if '_' in t:
                    tmp[tmp.index(t)] = '_'
                    answer.append(Answer[answerId])
                    answerId += 1
            article += ' '.join(tmp)
    if appendFlag == True or len(questions) == 0:
        questions.append({'article':article,'answer':answer,'option':option})

    return questions

def ParsePassageCompletion(context,name):
    context = [i for i in context if i != '']
    Answer = ans[name]
    answerId = 0
    questions = []
    context.pop(0)
    article = ''
    context.pop(0)
    option = []
    answer = []
    appendFlag = False
    for term in range(len(context)):
        tmp = context[term]
        tmp = tmp.split()
        key = isOption(context[term])
        if key:
            appendFlag = True
            index = indexer(tmp,optionType)
            for i in index:
                tmp[i[1]] = ''
            tmp = [i for i in tmp if i != '']
            option.extend(tmp)
        else:
            if appendFlag:
                appendFlag = False
                questions.append({'article':article,'answer':answer,'option':option})
                article = ''
                option = []
                answer = []
            for t in tmp:
                if '_' in t:
                    tmp[tmp.index(t)] = '_'
                    answer.append(Answer[answerId])
                    answerId += 1
            article += ' '.join(tmp)
    if appendFlag == True or len(questions) == 0:
        questions.append({'article':article,'answer':answer,'option':option})

    return questions

def ParseFt(context,name):   #TODO 從這裡開始
    context = [i for i in context if i != '']
    Answer = ans[name]
    print(context)
    questions = []
    context.pop(0)
    answerId = 0
    appendFlag = False
    article = ''
    option = []
    questionId = 0
    label = 0
    for term in range(len(context)):
        tmp = context[term]
        key = isOption(tmp)
        tmp = tmp.split()
        if key:
            appendFlag = True
            index = indexer(tmp,optionType)
            print(index)
            for i in index:
                tmp[i[1]] = ''
            tmp = [i for i in tmp if i != '']
            tmp = ' '.join(tmp)
            print(tmp)
            option.append(tmp)
        else:
            if appendFlag:
                print('why')
                appendFlag = False
                article = article.split()
                questionId = article.pop(0)
                article = ' '.join(article)
                label = ord(Answer[answerId]) - 65
                questions.append({'id':questionId,'context_sentence':article,'start_ending':'','ending_0': option[0],'ending_1': option[1],'ending_2': option[2],'ending_3': option[3],'label': label})
                option = []
                article = ''
                answerId += 1

            for t in tmp:
                if '_' in t:
                    tmp[tmp.index(t)] = '_'
            article += ' '.join(tmp)
    if appendFlag == True or len(questions) == 0:
        article = article.split()
        questionId = article.pop(0)
        article = ' '.join(article)
        label = ord(Answer[answerId]) - 65

        questions.append({'id':questionId,'context_sentence':article,'start_ending':'','ending_0': option[0],'ending_1': option[1],'ending_2': option[2],'ending_3': option[3],'label': label})

    return questions

def ParseRc(context,name):
    context = [i for i in context if i != '']
    Answer = ans[name]
    answerId = 0
    questions = []
    context.pop(0)
    article = ''
    #context.pop(0)
    option = []
    appendFlag = False
    start = '1'
    questionId = 0
    for term in range(len(context)):
        tmp = context[term]
        reIdx = retrieveIndex(tmp)
        key = isOption(tmp)
        if len(reIdx) > 0:
            start = reIdx[0]
        #tmp = tmp.split()
        if start+'.' in tmp:
            question = tmp
            question = question.replace(start+'.','')
            questionId = start
            start = str(int(start)+1)
        elif key:
            appendFlag = True
            index = indexer(tmp,optionType)
            for i in index:
                tmp[i[1]] = ''
            tmp = [i for i in tmp if i != '']
            tmp.pop(0)
            option.append(tmp)
        else:
            if appendFlag:
                appendFlag = False
                questions.append({'id':questionId,'article':article,'truth':ord(Answer[answerId])-65,'question':question,'option':option})
                article = ''
                option = []
                answerId += 1
            article += ' '.join(tmp)
    if appendFlag == True or len(questions) == 0:
        questions.append({'id':questionId,'article':article,'truth':ord(Answer[answerId])-65,'question':question,'option':option})
    return questions

def ParseTr(context,name):
    context = [i for i in context if i != '']
    vocAnswer = ans['Vocabulary']
    #print(context)
    questions = []
    context.pop(0)
    for term in range(len(context)):
        tmp = context[term]
        tmp = tmp.split()
        #print(tmp)
        for t in tmp:
            if '_' in t:
                #print(tmp.index(t))
                #print('---')
                tmp[tmp.index(t)] =  vocAnswer[term]
        questions.append({'sentence':' '.join(tmp),'answer':vocAnswer[term]})
    return questions

def main(filename):
    doc = docx.Document(filename)
    paragraphs = [para.text.lower() for para in doc.paragraphs]
    paragraphs = [i.replace(' ','') for i in paragraphs]
    index = [[i,paragraphs.index(i)]for i in questionType if i in paragraphs]
    paragraphs = [para.text for para in doc.paragraphs]
    index.append(['',len(paragraphs)-1])
    for idx in range(1,len(index)):
        tp = index[idx-1][0]
        qetype = paragraphs[index[idx-1][1]]
        qetype = ' '.join(qetype.split())
        parseArray = paragraphs[index[idx-1][1]:index[idx][1]]
        if tp == 'vocabulary':
            #ParseVoc(parseArray,qetype)
            pass
        elif tp == 'multiplechoice':
            #print(ParseMultiple(parseArray,qetype))
            pass
        elif tp == 'cloze':
            #print(ParseCloze(parseArray,qetype))
            pass
        elif tp == 'passagecompletion':
            pass
            #print(ParsePassageCompletion(parseArray,qetype))
        elif tp == 'readingcomprehension':
            print(ParseRc(parseArray,qetype))
            #pass
        elif tp == 'translation':
            #ParseTr(parseArray,qetype)
            pass
        elif tp == 'fitthebestsentence':
            #print(ParseFt(parseArray,qetype))
            pass
        elif tp == '':
            pass
        else:
            return 'error occured!'

def answer(filename):
    ans = dict()
    doc = docx.Document(filename)

    paragraphs = [para.text.lower() for para in doc.paragraphs]
    paragraphs = [i.replace(' ','') for i in paragraphs]
    index = [[i,paragraphs.index(i)]for i in questionType if i in paragraphs]
    paragraphs = [para.text for para in doc.paragraphs]
    index.append(['',len(paragraphs)-1])
    for idx in range(1,len(index)):
        tp = index[idx-1][0]
        qetype = paragraphs[index[idx-1][1]]
        parseArray = paragraphs[index[idx-1][1]:index[idx][1]]
        parseArray.pop(0)
        ans[qetype] = [pa.split('.')[0] for pa in parseArray]
    return ans
#main('exam.docx')
ans = answer('answerKey.docx')
print(ans)
main('exam.docx')

