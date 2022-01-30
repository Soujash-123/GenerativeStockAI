import requests
from bs4 import BeautifulSoup

def screener_scraper(company):
    resource = {}
    
    URL = f"https://www.screener.in/company/{company}/consolidated"
    html_text = requests.get(URL).text
    soup = BeautifulSoup(html_text , "lxml")
    values = soup.find_all("li", class_ = "flex flex-space-between")
    data = {}
    for value in values:
        name = value.find("span", class_ = "name").text.replace("\n", "").replace("  ", "")
        prop = value.find("span", class_ = "nowrap value").text.replace("\n", "").replace("  ", "")
        data[name] = prop
    
    resource['consolidated'] = data

    URL = f"https://www.screener.in/company/{company}/"
    html_text = requests.get(URL).text
    soup = BeautifulSoup(html_text , "lxml")
    values = soup.find_all("li", class_ = "flex flex-space-between")
    data = {}
    for value in values:
        name = value.find("span", class_ = "name").text.replace("\n", "").replace("  ", "")
        prop = value.find("span", class_ = "nowrap value").text.replace("\n", "").replace("  ", "")
        data[name] = prop

    resource['unconsolidated'] = data

    return resource
