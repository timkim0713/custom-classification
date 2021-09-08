
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os

cwd = os.getcwd()

#dir is not keyword


def makemydir(whatever):
    try:
        os.makedirs(whatever)
    except OSError:
        pass
    # let exception propagate if we just can't
    # cd into the specified directory
    os.chdir(whatever)


searchKey = input("What images do you want to search? \n ")
makemydir(searchKey)


options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(
    options=options, executable_path='/Users/timkim0713/Desktop/Others/chromedriver')


driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
elem = driver.find_element_by_name("q")
elem.send_keys(searchKey)
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1
for image in images:
    if count == 200:
        break
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element_by_xpath(
            '/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute("src")
        print("Downloading:   ", imgUrl)

        # opener = urllib.request.build_opener()
        # opener.addheaders = [
        #     ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        # urllib.request.install_opener(opener)
        urllib.request.urlretrieve(
            imgUrl,  cwd + "/" + searchKey + "/" + searchKey + "_" + str(count) + ".jpg")
        count = count + 1
    except:
        pass

driver.close()
