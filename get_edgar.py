import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

#will have to for-loop by year and quarter
year = 2019
qtr = 'QTR1'
edgarurl = "https://www.sec.gov/Archives/edgar/full-index/"+str(year)+"/"+str(qtr)+"/company.idx"

with urllib.request.urlopen(edgarurl) as url:
    data = url.readlines()

series = pd.Series(data)
df = pd.DataFrame(series)
df.columns = ['ABC']
df.ABC = df['ABC'].astype(str).str.replace("b'", "")
df.ABC = df['ABC'].astype(str).str.replace("\n", "")

'''
df.ABC = df['ABC'].astype(str).str.replace("  ", " ")

df.ABC = df['ABC'].astype(str).str.replace("||", "|")
df.ABC = df['ABC'].astype(str).str.replace("||", "|")
df.ABC = df['ABC'].astype(str).str.replace("||", "|")
df.ABC = df['ABC'].astype(str).str.replace("||", "|")

tenKs= df[df['ABC'].astype(str).str.contains("10-K")]
tenKs.columns =['col']
#tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("b'", "")
tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("   ", "  ")
tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("   ", "  ")
tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("   ", "  ")
tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("   ", "  ")
tenKs.loc['col'] = tenKs['col'].astype(str).str.replace("   ", "  ")
'''
df['A'], df['B'] = df['ABC'].str.split('  ', 1).str
df['B'], df['C'] =  df['B'].str.split('edgar/', 1).str

df['B'] = df['B'].astype(str).str.replace("  ", " ")
df['B'] = df['B'].astype(str).str.replace("  ", " ")
df['B'] = df['B'].astype(str).str.replace("  ", " ")
df['B'] = df['B'].astype(str).str.replace("  ", " ")


tenKs= df[df.B.astype(str).str.contains("10-K")]
tenQs = df[df.B.astype(str).str.contains("10-Q")]
############################3

#NOW We need to subset




example = https://www.sec.gov/Archives/edgar/data/1437517/0001493152-19-003191.txt
#create list of all the CIK codes....
edgarurl = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
   # "https://www.sec.gov/Archives/edgar/data/1326801/000132680118000067/fb-09302018x10q.htm"
with urllib.request.urlopen(edgarurl) as url:
    data = url.readlines()

data = pd.Series(data)
df = pd.DataFrame(data)
df.columns = ['ABC']
df['A'], df['B'], df['C'] = df['ABC'].astype(str).str.split(':', 1).str
#find a company
df[df['A'].str.contains("FACEBOOK")]










###GETTING THE SEC FILING
edgarurl = "https://www.sec.gov/Archives/edgar/data/1326801/000132680118000067/fb-09302018x10q.htm"
with urllib.request.urlopen(edgarurl) as url:
    html = url.readlines()
    soup = BeautifulSoup(html, features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

print(text)