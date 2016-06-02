'''Jan 2016 Aaron Tuor
Tool to scrape movielens 100k movie descriptions from imdb webpages
'''
import urllib.request
from urllib.request import Request, urlopen
import argparse
import sys
import re
import math
import os

#==================================================================
#=========BOOK KEEPING=============================================
#===================================================================
badurl = re.compile('[0-9]{6}$')
# windows compliant
slash = '/'
if os.name == 'nt':
    slash = '\\'  # so this works in Windows
parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str)
parser.add_argument('-outpath', type =str, default='')
args = parser.parse_args()

# doesn't matter if you forget or add a slash
if not args.datapath.endswith(slash):
	args.datapath += slash
if not args.outpath.endswith(slash) and not args.outpath == '':
	args.outpath += slash
#======================================================================

def get_text(url):
	req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	webpage = urlopen(req)

	line = webpage.readline().decode('utf-8')
	description = ''
	#if line.strip.startswith('<link rel="canonical"'):
	con = True
	while not line.strip().startswith('<meta name="description" content="'):
		line = webpage.readline().decode('utf-8')
		if line.strip().startswith('</html>'):
			break
		if line.strip().startswith('<link rel="canonical" href="http://www.imdb.com/find"'):
			con = False
			break
	if con:
		description += line.split('"')[3]
		while not line.strip().startswith('<div class="inline canwrap" itemprop="description">'):
			line = webpage.readline().decode('utf-8')
			if line.strip().startswith('</html>'):
				break
		webpage.readline()
		description += ' ' + webpage.readline().decode('utf-8').split('   ')[0]
		return description
	else:
		while not line.strip().startswith('<tr class="findResult odd">'):
			line = webpage.readline().decode('utf-8')
		linesection = line.split('<')[3].split('"')[1].strip()
		webpage.close()
		description = get_text('http://www.imdb.com/'+linesection)
	return description

#=====================================================================
#==============MAIN===================================================
#=====================================================================
with open(args.outpath + 'descriptions.txt', 'w') as descriptions:
	with open(args.datapath + 'u.item', encoding='latin-1') as infile:
		lines = infile.readlines()
		# ind = 1
		for line in lines:
			print(line)
			url = line.split('|')[4].strip()
			if line.split('|')[3] == 'unknown':
				descriptions.write('none\n')
			elif re.match('[0-9]{6}$', url.split('-')[-1]):
				url = "http://www.imdb.com/title/tt" + url.split('-')[-1] + "/?ref_=fn_al_tt_1" 
				print(url)
				description = get_text(url)
				descriptions.write(description + '\n')
			else:
				print(url)
				description = get_text(url)
				descriptions.write(description + '\n')
			# ind += 1

