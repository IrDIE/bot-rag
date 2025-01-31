import requests
from bs4 import BeautifulSoup

def get_news_links(url, base):
    page = requests.get(url)
    bSoup = BeautifulSoup(page.content, 'html.parser')
    links_list = bSoup.find_all('a')

    #filter links for articles:
    article_links_all = [
        base + l.get('href') for l  in links_list 
        if 'href' in l.attrs and '/news/' in l.attrs['href']
    ]

    # get rid of duplicates:
    return list(set(article_links_all))

def parce_news_sublinks(url):
    # 
    urls = [f"{url}{i}" for i in range (2,6)]
    base = "https://news.itmo.ru"

    all_mews = get_news_links(url, base)
    for url in urls:
        all_mews += get_news_links(url, base)
    return all_mews



if __name__=="__main__":
    
    mainpage_urls = [
        "https://news.itmo.ru/ru/main_news/",
        "https://news.itmo.ru/ru/science/life_science/",
        "https://news.itmo.ru/ru/science/new_materials/",
        "https://news.itmo.ru/ru/science/cyberphysics/",
        "https://news.itmo.ru/ru/science/photonics/",
        "https://news.itmo.ru/ru/science/it/",
        "https://news.itmo.ru/ru/education/official/",
        "https://news.itmo.ru/ru/education/students/",
        "https://news.itmo.ru/ru/education/trend/",
        "https://news.itmo.ru/ru/education/cooperation/",
        "https://news.itmo.ru/ru/startups_and_business/business_success/",
        "https://news.itmo.ru/ru/startups_and_business/innovations/",
        "https://news.itmo.ru/ru/startups_and_business/startup/",
        "https://news.itmo.ru/ru/startups_and_business/partnership/",
        "https://news.itmo.ru/ru/startups_and_business/initiative/",
        "https://news.itmo.ru/ru/university_live/ratings/",
        "https://news.itmo.ru/ru/university_live/achievements/",
        "https://news.itmo.ru/ru/university_live/leisure/",
        "https://news.itmo.ru/ru/university_live/ads/",
        "https://news.itmo.ru/ru/university_live/social_activity/"


        ]

    all_all_news = []
    for page_url in mainpage_urls:
        all_mews = parce_news_sublinks(page_url)
        print(all_mews)
        all_all_news += all_mews