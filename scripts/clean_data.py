import sys
import nltk

def output(lines, f2):
	o_line = ''
	for line in lines:
		o_line = o_line + line + '\t'
	print(o_line[:-1], file=f2, end='')

with open(sys.argv[1]) as f:
	f2 = open(sys.argv[1].split('/')[-1], 'w')
	l = f.readlines()
	# maxi = 0
	# count = 0
	for line in l:
		parts = line.split('\t')
		name = parts[4].split('(')[0].strip().lower()
		name_length = len(name.split(' '))
		if name_length > 0 and name_length <= 6:
			wiki_title = (' '.join([x.lower() for x in parts[0].split('_')])).split('(')[0].strip()
			entities = [x.strip() for x in parts[1].strip().split(';')]
			if wiki_title not in entities:
				min_dist = 100000000
				wiki_title2 = wiki_title
				for ent in entities:
					dst = nltk.edit_distance(ent, wiki_title)
					# print(ent, dst)
					if dst < min_dist:
						min_dist = dst
						wiki_title2 = ent
				# print(wiki_title, ' : ' , wiki_title2)
				# print(entities)
				# print(name)
				# print('')
				wiki_title = wiki_title2

			index = entities.index(wiki_title)
			garbage = '<garbage_' + str(index) + '>'
			right = [x.strip() for x in parts[4].split('(')]
			right = (' '.join(right[1:])).split(')')
			right = [x.strip() for x in right]
			right = right[1:]
			full = garbage + ' ' + ' '.join(right)
			parts[4] = full
			parts[0] = wiki_title
			output(parts, f2)
		else:
			# print(name)
			output(parts, f2)
