from asyncio.windows_events import NULL
from asyncore import read
from audioop import add
import csv
import json
from lib2to3.pgen2 import driver
from time import sleep, time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import json

fieldnames = ['Area', 'Restaurant Name', 'Image', "Address", "Contact", "Dinning Rating", "Number of dinning rating",
              "Delivery Rating", "Number of delivery rating", "Cuisines", "Opening Time", "Opening Days", "People Say",
              "Average Cost", "Menu Card", "Payment Modes", "Lattitude", "Longitude", ]

with open("data.csv", "a") as file:
    fieldnames = ['Area', 'Restaurant Name', 'Image', "Address", "Contact", "Dinning Rating", "Number of dinning rating",
                  "Delivery Rating", "Number of delivery rating", "Cuisines", "Opening Time", "Opening Days", "People Say",
                  "Average Cost", "Menu Card", "Payment Modes", "Lattitude", "Longitude", ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

def extractData(url, file, area):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
        "Accept": "text/html,/;q=0"}
    # url = "https://www.zomato.com/pune/mcdonalds-kharadi"
    url = "https://www.zomato.com" + url
    response = requests.get(url, headers=headers).content
    soup = BeautifulSoup(response, "html.parser")

    name = soup.find("h1", {"class": "sc-7kepeu-0 sc-kafWEX kTxZkY"}).text
    diningRating = soup.find("div", {"class": "sc-1q7bklc-1 cILgox"}).text
    noOfDiningRating = soup.find("div", {"class": "sc-1q7bklc-8 kEgyiI"}).text
    deliveryRating = (soup.find_all(
        "div", {"class": "sc-1q7bklc-1 cILgox"}, limit=2))[1].text
    noOfDeliveryRating = (soup.find_all(
        "div", {"class": "sc-1q7bklc-8 kEgyiI"}, limit=2))[1].text
    cuisine = soup.find_all("a", {"class": "sc-ibxdXY cgmKgO"})

    for i in range(len(cuisine)):
        cuisine[i] = cuisine[i].text
    soup = soup.prettify()
    data = soup.split("PRELOADED_STATE__ = JSON.parse(")[1]
    data = data.split(");")[0]

    # data = json.loads(data)
    # data

    try:
        address = ((json.loads(data).split("address\":\"")[1]).split("\",")[0])
    except:
        address = NULL

    try:
        phone = ((json.loads(data).split("phoneStr\":\"")[1]).split("\"},")[0])
    except:
        phone = NULL

    try:
        opening = ((json.loads(data).split("{\"timing\":\"")[1]).split("\"}]}")[
                   0]).split("\",\"days\":\"")
        openTiming = opening[0]
        openDays = opening[1]
    except:
        opening = NULL
        openTiming = NULL
        openDays = NULL

    try:
        menu = ((json.loads(data).split("IMAGE_MENUS")[
            1]).split("thumb\":\"")[1]).split("?")[0]
    except:
        menu = NULL

    try:
        peopleSay = ((json.loads(data).split(
            "People Say This Place Is Known For\",\"description\":\"")[1]).split("\"}")[0])
    except:
        peopleSay = NULL

    try:
        avgCostFor2People = (
            (json.loads(data).split("₹")[1]).split("(approx")[0])
    except:
        avgCostFor2People = NULL

    try:
        modesOfPayment = ((json.loads(data).split(
            "accepted_payments\":\"")[1]).split("\"},")[0])
    except:
        modesOfPayment = NULL

    try:
        latitude = ((json.loads(data).split(
            "latitude\":\"")[1]).split("\",")[0])
    except:
        latitude = NULL

    try:
        longitude = ((json.loads(data).split(
            "longitude\":\"")[1]).split("\",")[0])
    except:
        longitude = NULL

    try:
        image = (((json.loads(data).split("\"IMAGES\":")[1]).split(
            "\"url\":\"")[1]).split("\",\""))[0]
    except:
        image = NULL

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    try:
        writer.writerow({'Area': area, 'Restaurant Name': name, 'Image': image, "Address": address, "Contact": phone,
                         "Dinning Rating": diningRating, "Number of dinning rating": noOfDiningRating, "Delivery Rating": deliveryRating,
                         "Number of delivery rating": noOfDeliveryRating, "Cuisines": cuisine, "Opening Time": openTiming, "Opening Days": openDays,
                         "People Say": peopleSay, "Average Cost": avgCostFor2People, "Menu Card": menu, "Payment Modes": modesOfPayment, "Lattitude": latitude,
                         "Longitude": longitude})
    except Exception as ex:
        print(ex)

    # print()
    # print("Restaurant Name:", name, "\nimage:", image, "\nAddress:", address, "\ncontact:", phone,
    #       "\ndining rating: ", diningRating, "\nNo Of dining Rating: ", noOfDiningRating,
    #       "\ndelivery rating: ", deliveryRating, "\nNo Of delivery Rating: ", noOfDeliveryRating,
    #       "\ncuisines: ", cuisine, "\nopening time: ", openTiming, "\nopening days: ",
    #       openDays, "\nWhat People say: ", peopleSay,
    #       "\naverage cost for two: ", avgCostFor2People, "\nMenu card:", menu,
    #       "\nModes of payment", modesOfPayment, "\nlatitude and longitude: ", latitude, longitude)
    # print()


def extractHotelLinks():
    hotelLinkFile = open("hotelLinks.txt", "a")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0',
        "Accept": "text/html,*/*;q=0"}
    response = requests.get(
        "https://www.zomato.com/pune", headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")
    area = soup.find_all('h5')
    req = []
    for i in range(len(area)):
        area[i] = (str(area[i]).split(" (")[0]).lower().replace(" ", "-")
        area[i] = area[i].rsplit(">")[1]
    for a in area:
        req.append("https://www.zomato.com/pune/"+a+"-restaurants")


    option = webdriver.ChromeOptions()
    chrome_prefs = {}
    option.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}

    sleepTime = 1

    driver = webdriver.Chrome(
        ChromeDriverManager().install(), chrome_options=option)

    count = 0
    for areaLink in req:
        driver.get(areaLink)
        screenHeight = driver.execute_script("return window.screen.height;")
        currHeight = 0

        hotelLinks = set()

        soup = BeautifulSoup(driver.page_source, "html.parser")
        # soup = soup.prettify()
        hotels = soup.findAll("h4")
        # hotelLinks = soup.find_all("a", {"class": "sc-jsqbLq gstmgm"})
        # print(hotelLinks)
        for hotel in hotels:
            hotelLinks.add(hotel.parent.parent["href"].rsplit("/", 1)[0])

        # print(str(hotelLinks.pop()))
        # extractData(str(hotelLinks.pop()))

        pageCount = 0
        while True:
            driver.execute_script("window.scrollTo(0, {0});".format(currHeight))
            # if pageCount == 3:
            #     break
            # pageCount += 1

            maxHeight = driver.execute_script("return document.body.scrollHeight;")

            if(currHeight > maxHeight):
                break
            currHeight += screenHeight
            print(currHeight, maxHeight)
            if(maxHeight >= currHeight + 5 * screenHeight):
                if(sleepTime < 0.001):
                    sleepTime = 0
                else:
                    sleepTime /= 2
            else:
                sleepTime = 2
            print("sleep time: ", sleepTime)
            sleep(sleepTime)

        print("fetched " + area[count] + " html page")
        # type(driver.page_source)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        Links = soup.find_all("a", {"class": "sc-hdPSEv keaTne"})
        for link in Links:
            res = (link["href"]).rsplit("/", 1)[0]
            hotelLinks.add(res)
            hotelLinkFile.write(area[count] + " " + res + "\n")

        print("extracted " + area[count] + " links")

        # file = open("data.csv", "a", newline="")
        # for link in hotelLinks:
        #     extractData(str(link), file, area[count])
        # print("done: " + area[count])

        count += 1


def extractDataFromHotelLinks():
    fileN = open("hotelLinks.txt", "r")
    file = open("data.csv", "a", newline="")
    recordCnt = 1
    for line in fileN:
        locality = line.split(" ")[0]
        link = line.split(" ")[1].split("\n")[0]
        try:
            extractData(link, file, locality)
        except Exception as ex:
            print(ex)
        print(recordCnt)
        recordCnt += 1

# # Call this function first to extract hotelLinks
# extractHotelLinks()

# Call this function if hotelLinks are available in hotelLinks.txt file
extractDataFromHotelLinks()