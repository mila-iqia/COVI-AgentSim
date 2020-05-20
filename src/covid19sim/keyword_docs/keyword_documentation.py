import os
import shutil
import re

# Pattern for regexing keywords
pattern = '(&[\S]+)\s+\[\s?(.*)\]'

def get_keyword_dict_from_src (src_path='src'):
	# Loop through all python files that have parameters to be documented
	python_files = [f for f in os.listdir() if f.endswith('py')] 
	keywords = []
	citations = []
	for filename in python_files:
		with open(filename, 'r') as source_file:
			print ('\n'+filename)
			for line in source_file:
				# Check if line contains a keyword
				match = re.search(pattern)
				if match is not None: 
					keyword = match.group(1)
					citation = match.group(2)
					keywords.append(keyword)
					citations.append(citation)
	keyword_dict = dict(zip(keywords, citations))
	return keyword_dict


def get_keyword_dict_from_spreadsheet (spreadsheet_path=''):
	# Connect to the google spreadsheet
	# modified from https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html?utm_source=youtube&utm_medium=video&utm_campaign=youtube_python_google_sheets
	import gspread
	from oauth2client.service_account import ServiceAccountCredentials

	scope = ['https://www.googleapis.com/auth/drive']
	creds = ServiceAccountCredentials.from_json_keyfile_name('covi_documentation_auth.json', scope)
	client = gspread.authorize(creds)
	sheet = client.open("Covi Parameter Documentation").sheet1

	# Make a dict of keywords and citations
	keywords = sheet.col_values(5)
	keys = keywords[1:]
	citations = sheet.col_values(6)
	values = citations[1:]
	keyword_dict = dict(zip(keys, values))
	return keyword_dict
	
def add_keywords_to_spreadsheet():
	# Connect to the google spreadsheet
	# modified from https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html?utm_source=youtube&utm_medium=video&utm_campaign=youtube_python_google_sheets
	import gspread
	from oauth2client.service_account import ServiceAccountCredentials

	scope = ['https://www.googleapis.com/auth/drive']
	creds = ServiceAccountCredentials.from_json_keyfile_name('covi_documentation_auth.json', scope)
	client = gspread.authorize(creds)
	sheet = client.open("Covi Parameter Documentation").sheet1

	# Loop through all python files that have parameters to be documented
	python_files = ['config.py'] #[f for f in os.listdir() if f.endswith('py')] 
	for filename in python_files:
		with open(filename, 'r') as source_file:
			print ('\n'+filename)
			with open ('outfile.py', 'w') as out_file:
				for line in source_file:

					# Check if line contains a keyword
					match = re.search(pattern, line)
					if match is not None: # Parse what we can into the spreadsheet

						# Keyword and Citation
						keyword = match.group(1)
						citation = match.group(2)

						# Name, variable, and Value
						data = line.split('=')
						if len(data) == 1:
							data = line.split(':')
						if len(data) == 1:  # no equals sign or colon
							variable = line.split('#')[0]
							value = '?'
						else:
							variable = data[0].strip()
							value = data[1].split('#')[0].strip().strip(',')
						name = variable.lower().replace('_', ' ')
						
						# Statistic
						if 'avg' in name or 'mean' in name:
							stat = 'average'
						elif 'scale' in name or 'std' in name:
							stat = 'standard deviation'
						elif 'median' in name:
							stat = median
						else:
							stat = ''

						# Units
						comment = line.split('#')[1]
						units = comment.split('&')[0].strip()
						if 'days' in name:
							units = 'days'
						elif 'hours' in name:
							units = 'hours'

						# Notes
						after_citation = comment.split(']')
						if len(after_citation) > 1:
							notes = after_citation[1].strip()

						# Build row and insert in spreadsheet
						row = [name,stat,value,units,keyword,citation,notes]
						out_file.write('\t'.join(row)+'\n')
						#sheet.insert_row(row, 2)


def update_comments_in_src(src_path='src', spreadsheet_path=''):
	keyword_dict = get_keyword_dict_from_spreadsheet()
	python_files = [f for f in os.listdir() if f.endswith('py')] 
	num_updated_total = 0
	for filename in python_files:
		num_updated_file = 0
		edits_required = False
		tmp_filename = os.join(filename, '.tmp')
		with open(filename, 'r') as source_file:
			print ('\n'+filename)
			with open(tmp_filename, 'w'):
				for line in source_file:
					match = re.search(pattern, line)
					if match is not None:
						keyword = match.group(1)
						citation = match.group(2)
						if not keyword in keyword_dict:
							print ('WARNING: keyword '+ keyword + ' not found in spreadsheet.')
						    break
						if citation == keyword_dict[keyword]:
							print ('[OK] '+ keyword + ' ['+ citation +']')
						else:
							edits_required = True
							num_updated_file += 1
							num_updated_total += 1
							citation = keyword_dict[keyword]
							print ('[UP] '+ keyword + ' ['+ citation +']')
		if edits_required:
			shutil.move(src=tmp_filename, dst=filename)
		print (str(num_updated_file) + 'edits made to '+ filename)
	print (str(num_updated_total) + 'edits made in total.'

