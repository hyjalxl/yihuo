# coding=utf-8
# user=hu

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver import ActionChains

import time
import re


def get_driver():
    firefox_option = Options()
    firefox_option.add_argument('--headless')
    driver = webdriver.Firefox(options=firefox_option)
    return driver

def get_url(driver, url):
    driver.get(url)


if __name__ == "__main__":
    url = 'https://h5api.m.taobao.com/h5/mtop.taobao.detail.getdetail/6.0/?jsv=2.4.8&appKey=12574478&t=1514363050271&sign=58c772297bca044c441667acae13e869&api=mtop.taobao.detail.getdetail&v=6.0&ttid=2016%40taobao_h5_2.0.0&isSec=0&ecode=0&AntiFlood=true&AntiCreep=true&H5Request=true&type=jsonp&dataType=jsonp&callback=mtopjsonp1&data=%7B%22exParams%22%3A%22%7B%5C%22id%5C%22%3A%5C%22543736245887%5C%22%2C%5C%22abtest%5C%22%3A%5C%222%5C%22%2C%5C%22rn%5C%22%3A%5C%2214f65a30b76082ef68a59833849b2b8f%5C%22%2C%5C%22sid%5C%22%3A%5C%2242b9f87ba47a92e8b98e1621576c32dd%5C%22%7D%22%2C%22itemNumId%22%3A%22543736245887%22%7D'
    driver = get_driver()
    sliding_url = get_url(driver, url)


    driver.quit()
