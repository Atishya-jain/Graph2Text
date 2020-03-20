with open('../GraphWriter/data/cur_data.train.tsv') as f:
	f2 = open('train.tsv', 'w')
	l = f.readlines()
	maxi = 0
	count = 0
	for line in l:
		parts = line.split('\t')
		length = len(parts[3].split(';'))
		if length > 200:
			count += 1
		else:
			print(line, file=f2, end='')

		if length >= maxi:
			maxi = length

	print(count)
	print(maxi)
		# break
