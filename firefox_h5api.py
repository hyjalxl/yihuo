# coding=utf-8
# user=hu

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
# webDriverWait 库，负责循环等待
from selenium.webdriver.support.ui import WebDriverWait
# expected_conditions 累负责条件触发
from selenium.webdriver.support import expected_conditions as EC

import re
import time
import urllib2
import requests
import firefox_headless


def send_cookie(cookie_value):
    """
    向公司json服务器发送cookie
    :param cookie_value:
    :return:
    """
    url = 'https://api.xiaoyataoke.com/api/XiaoYaTaoKe/AddV6Cookie?shop_name=ZzZrdWd1Vk8zWXFISHM2TVNlMlFKRlNLL3ZjWXlaTTU=&data_source=huyangjie&cookies=' + cookie_value
    # print url
    headless_firefox_driver = firefox_headless.get_driver()
    headless_firefox_driver.get(url)
    headless_firefox_driver.save_screenshot('response' + str(time.time())[3:10] + '.png')
    # print 'response is:', headless_firefox_driver.page_source
    headless_firefox_driver.quit()


def get_sliding_url(url):
    print 'get_sliding_url'
    header = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    request = urllib2.Request(url, headers=header)
    for i in range(10):
        try:
            response = urllib2.urlopen(request)
            html = response.read()
            print len(html)
            if len(html) < 10000:
                # print html
                return re.findall(ur'https://.*?com/', html, re.I)[0]
            time.sleep(0.1)
        except:
            print 'Get_sliding_url is wrong!'


def sliding(sliding_url):
    """
    滑动函数，
    :param sliding_url: 需要滑动的网页
    :return:
    """
    driver = webdriver.Firefox()
    driver.delete_all_cookies()
    driver.get(sliding_url)
    try:
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '//span[@id="nc_1_n1z"]'))
        )
    except:
        print 'Not find element'
    dr1 = ActionChains(driver)
    dr1.click_and_hold(element).pause(1)
    dr1.move_by_offset(100, 0).pause(1).move_by_offset(160, 0)
    dr1.perform()
    time.sleep(1)
    for cookie in driver.get_cookies():
        # print cookie
        if cookie['name'] == 'x5sec':
            # print '%s -> %s' % (cookie['name'], cookie['value'])
            cookie_value = cookie['value']

    driver.quit()
    time.sleep(1)
    send_cookie(cookie_value)


def dispatch():
    """
    调度函数负责调度各函数
    :return:
    """
    url = 'https://h5api.m.taobao.com/h5/mtop.taobao.detail.' \
          'getdetail/6.0/?jsv=2.4.8' \
          '&appKey=12574478' \
          '&t=1514363050271' \
          '&sign=58c772297bca044c441667acae13e869' \
          '&api=mtop.taobao.detail.getdetail&v=6.0' \
          '&ttid=2016%40taobao_h5_2.0.0' \
          '&isSec=0&ecode=0&AntiFlood=true' \
          '&AntiCreep=true' \
          '&H5Request=true&type=jsonp' \
          '&dataType=jsonp&callback=mtopjsonp1' \
          '&data=%7B%22exParams%22%3A%22%7B%5C%22id%5C%22%3A%5C%22543736245887%5C%22%2C%5C%22abtest%5C%22%3A%5C%222%5C%22%2C%5C%22rn%5C%22%3A%5C%2214f65a30b76082ef68a59833849b2b8f%5C%22%2C%5C%22sid%5C%22%3A%5C%2242b9f87ba47a92e8b98e1621576c32dd%5C%22%7D%22%2C%22itemNumId%22%3A%22543736245887%22%7D' \
          '&qq-pf-to=pcqq.c2c'
    print url
    sliding_url = get_sliding_url(url)
    # print sliding_url
    if sliding_url:
        sliding(sliding_url)


if __name__ == "__main__":

        for i in range(1, 200000):
            try:
                print time.asctime()
                dispatch()
                for h in range(10):
                    print '第%s次测试，再等%s分钟下一次测试。' %(i, (10-h))
                    time.sleep(60)
            except:
                print '小乖乖出错了！她会再次运行。'