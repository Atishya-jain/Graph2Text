import wikipedia
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

f = open("../fb15k/entity_mid_name_type_typeid.txt", "r")
to_run_on = []
for i in f.readlines():
	a,b,c,d = i.strip("\n").split("\t")
	if c == 'people':
		to_run_on.append([(a,0,[])])
f.close()

to_run_on = [[("/m/025n3p",0,[])]]
def wiki_extract(title):	
	title = " ".join(title.split('_'))
	ny = wikipedia.page(title)
	text = (' '.join(ny.summary.split('\n')))
	tokenized_text = word_tokenize(text)
	full_text = ' '.join(tokenized_text).lower()
	return full_text.split('.')

class Graph:
	def __init__(self):
		edges = {}
		for relations in ["../fb15k/train.txt"]:
			f = open(relations, "r")
			for i in f.readlines():
				e1,r,e2 = i.strip("\n").split("\t")
				if e1 not in edges:
					edges[e1] = [(e2,r)]
				else:
					edges[e1].append((e2,r))
				if e2 not in edges:
					edges[e2] = [(e1,r)]
				else:
					edges[e2].append((e1,r))
			f.close()
		self.edges = edges

def graph(q, fbgraph):
	f = open("../fb15k/mid2wikipedia_cleaned.tsv", "r")
	edges = fbgraph.edges
	mid2wiki = {}
	for i in f.readlines():
		a,b,c = i.strip("\n").split("\t")
		mid2wiki[a] = (b,c)
	f.close()
	my_name = mid2wiki[q[0][0]][1]
	DEPTH = 2
	entities = []
	while len(q) > 0:
		x,d,l = q.pop(0)
		entities.append((x,l))
		if (d < DEPTH):
			for (e,r) in edges[x]:
				q.append((e,d+1,list(l) + [(x,r)]))
	graph_ents = []
	for (entity,l) in entities:
		graph_ents.append((mid2wiki[entity][0].lower(), [(mid2wiki[e][0].lower(), r) for (e,r) in l]))
	return list((graph_ents)),my_name

iteration = 0
count = 0
total = 0
data = []
f = open("cur_data", "w")
fbgraph = Graph()
for q in to_run_on:
	# if iteration%10 == 1:
		# print("-----------------------------------------", iteration)
		# print("Match percent:", (count/total)*100)
	iteration += 1
	graph_ents, my_name = graph(q, fbgraph)
	sentences = wiki_extract(my_name)
	# print(graph_ents)
	final_sentence = []
	selected_ents = []
	total_characters = 0
	# print(sentences)
	our_graph_edges = []
	for i in sentences:
		flag = False
		total_characters += len(i)
		for j in graph_ents:
			if " " + j[0] + " " in i:
				if not flag:
					final_sentence.append(i.strip())
					flag = True
				selected_ents.append(j[0])
				assert len(j[1]) <= 2
				if (len(j[1]) == 1):
					our_graph_edges.append((j[1][0][0],j[1][0][1],j[0]))
				elif (len(j[1]) == 2):
					our_graph_edges.append((j[1][0][0],j[1][0][1],j[0]))
					our_graph_edges.append((j[1][1][0],j[1][1][1],j[1][0][0]))
				# break
		# if total_characters >= 500:
		# 	break
	# print (our_graph_edges)
	final_sentence = ' '.join(final_sentence)
	# data.append((my_name, set(selected_ents), final_sentence))
	print(set(selected_ents))
	print(final_sentence)
	f.write("\n".join([str(i) for i in list(set(our_graph_edges))]))
	f.write(" ".join(list(set(selected_ents))) + "\t" + final_sentence + "\n")
	# print((my_name, set(selected_ents), final_sentence))
	# total += len(final_sentence.split())/2
	# for i in graph_ents:
	# 	if i in final_sentence:
	# 		count += 1	
f.close()
