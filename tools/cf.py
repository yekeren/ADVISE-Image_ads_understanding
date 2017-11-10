import json

sets = {}
with open('output/visual_words.text', 'r') as fp:
  for line in fp.readlines():
    words = set()
    word_id, words1, words2 = line.strip('\n').split('\t')
    if words1: words.update(words1.split(','))
    if words2: words.update(words2.split(','))
    sets[int(word_id)] = words

with open('output/visual_words.json', 'r') as fp:
  data = json.loads(fp.read())

co_occur = {}
occur = {}

for _, items in data.iteritems():
  items = sorted(set(items))
  for i in xrange(len(items)):
    occur[items[i]] = occur.get(items[i], 0) + 1
    for j in xrange(i + 1, len(items)):
      co_occur[(items[i], items[j])] = co_occur.get((items[i], items[j]), 0) + 1
      co_occur[(items[j], items[i])] = co_occur.get((items[j], items[i]), 0) + 1

results = []
for k, v in co_occur.iteritems():
  item_i, item_j = k
  similarity = 1.0 * v / occur[item_i]

  text_similarity = 1.0 * len(sets[item_i] & sets[item_j]) / len(sets[item_i] | sets[item_j])
  results.append((item_i, item_j, similarity, text_similarity))

results = filter(lambda x: x[2] >= 0.20, results)
results = sorted(results, lambda x, y: cmp(x[3], y[3]))

html = ''
html += '<table border=1>'
html += '<tr>'
html += '<th>visual word similarity</th>'
html += '<th>surface word similarity</th>'
html += '<th>visual word 1</th>'
html += '<th>visual word 2</th>'
html += '</tr>'
for item_i, item_j, similarity, text_similarity in results:
  html += '<tr>'
  html += '<td>%.4lf</td>' % (similarity)
  html += '<td>%.4lf</td>' % (text_similarity)
  html += '<td><a href="visual_words.html#%d">word_%d</a></td>' % (item_i, item_i)
  html += '<td><a href="visual_words.html#%d">word_%d</a></td>' % (item_j, item_j)
  html += '</tr>'
html += '</table>'

with open('output/co-occur.html', 'w') as fp:
  fp.write(html)
