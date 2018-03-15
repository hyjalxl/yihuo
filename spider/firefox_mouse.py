# coding=utf-8
# user=hu

from selenium import webdriver
from selenium.webdriver import ActionChains
import time


def sliding_mouse():

    driver = webdriver.Firefox()
    driver.get('http://localhost/mouse.html')
    dr1 = ActionChains(driver)
    time.sleep(3)
    dr1.click_and_hold(driver.find_element_by_id('spliding')).release()
    # for i in range(7):
    #     dr1.move_by_offset(30, 50)
    #     dr1.move_by_offset(10, -30)
    dr1.perform()
    time.sleep(9)
    driver.quit()

if __name__ == "__main__":
    sliding_mouse()
