from bs4 import BeautifulSoup

# Assuming 'combined.html' is the file you created with 'cat'
with open('combined.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser') # Or 'lxml' for better parsing

# Get all visible text from the entire document
all_text = soup.get_text(separator='\n', strip=True)

print(all_text)