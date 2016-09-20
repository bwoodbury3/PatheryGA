import urllib2

def sendPath(mapcode, mapID, solution, isChallenge=False):
    regStr = "isChallenge=" + str(isChallenge).lower()
    regStr += "&r=getpath"
    regStr += "&mapcode=" + mapcode
    regStr += "&mapID=" + str(mapID)
    regStr += "&solution=" + solution
    request = urllib2.Request("http://pathery.com/do.php?" + regStr)
    request.add_header("Cookie", "__cfduid=d8db4e1af331cdd372426c1c3ce5ee50d1469555253; __utma=80352790.1579500229.1469555255.1471728920.1471791632.29; __utmz=80352790.1469555255.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); pref_speed=5; pref_mute=true; PHPSESSID=6mc0vsnikg6rkf7d9b6a7su455; __utmb=80352790.10.10.1471791632; __utmc=80352790; userID=797; doLogin=yes; auth=a526d1bed018fa9352ded0884bc40e08; mp_24743c6567f831ddfcbbbd3f397e11e4_mixpanel=%7B%22distinct_id%22%3A%20%22797%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%2C%22__mps%22%3A%20%7B%7D%2C%22__mpa%22%3A%20%7B%7D%2C%22__mpap%22%3A%20%5B%5D%2C%22%24people_distinct_id%22%3A%20%22797%22%2C%22mp_name_tag%22%3A%20%22B%22%7D")
    request.add_header("Connection", "keep-alive")
    raw = urllib2.urlopen(request).read()
