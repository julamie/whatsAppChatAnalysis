import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# replace zero width character with empty character
def removeZeroWidthSpace(data: str):
    return data.replace(u'\u200b', "")

# escape special characters with \ to prevent messing with the data when converting
def escapeSpecialChars(data: str):
    data = data.replace("\\", r"\\")
    data = data.replace("\"", r"\"")
    data = data.replace("\'", r"\'")
    data = data.replace("|", r"\|")
    return data

# edit format of file to csv, add quotation marks around messages and end message with |
def convertTextToCSVFormat(data: str):
    # Yes, I know this looks cursed
    data = re.sub(r"(\d\d.\d\d.\d\d), (\d\d:\d\d) - ([\w\s]+): (.*)\n", "\"|\\n\\1, \\2, \\3, \"\\4", data)

    # because of the nature of this regex, there is a "| too much at the beginning
    # and a "| too little at the end
    data = data[3:] # remove ", | and \n
    data = data + '\"|' # add " and \n

    # add names for columns at beginning of file
    data = "Date,Time,Name,Message|\n" + data

    return data

# opens file, clears it from weird characters and saves result in new file
def preprocessFile(oldPath: str, newPath: str):
    with open(oldPath, encoding="utf-8") as origFile:
        with open(newPath, 'w', encoding="utf-8") as newFile:
            data = origFile.read()
            data = removeZeroWidthSpace(data)
            data = escapeSpecialChars(data)
            data = convertTextToCSVFormat(data)
            newFile.write(data)

# converts our preprocessed file into a pandas dataframe
def convertFileToDataframe(path: str):
    df = pd.read_csv(path, usecols=range(4), skipinitialspace=True, lineterminator="|", encoding="utf-8")

    # show full length of message when printing
    pd.set_option("max_colwidth", None)
    pd.set_option("max_seq_item", None)

    return df

# removes the beginning \r\n from the dates
def removeBeginningSpecialChars(df: pd.DataFrame):
    df["Date"] = df["Date"].replace(r"\r\n", "", regex=True)
    return df

# removes the \ from escaped chars in message
def revertEscapedChars(df: pd.DataFrame):
    # I don't exactly know why this one works, but it works
    df["Message"] = df["Message"].replace(r"\\\\", "\\\\", regex=True)
    df["Message"] = df["Message"].replace(r"\"", "\"", regex=True)
    df["Message"] = df["Message"].replace(r"\\'", "\'", regex=True)
    df["Message"] = df["Message"].replace(r"\|", "|", regex=True)

# adds length of message as a column to dataframe
def addColumnMessageLength(df: pd.DataFrame):
    df["Message length"] = df["Message"].str.len()

# adds number of words in message as a column
def addColumnNumberOfWords(df: pd.DataFrame):
    df["Message word count"] = df["Message"].apply(lambda x: len(str(x).split(' ')))

# cleans the dataframe after creation
def postprocessData(df: pd.DataFrame):
    # clean data
    removeBeginningSpecialChars(df)
    revertEscapedChars(df)

    # add new data
    addColumnMessageLength(df)
    addColumnNumberOfWords(df)
    return df

# count how many messages everyone sent
def countMessagesByName(df: pd.DataFrame):
    print("Sent messages by user")
    print(df.groupby("Name") \
        .count() \
        .sort_values(by="Message", ascending=False)["Message"])
    print()

# count how many words everyone sent
def countWordsByName(df: pd.DataFrame):
    print("Sent words by user")
    print(df.groupby("Name") \
        .sum() \
        .sort_values(by="Message word count", ascending=False)["Message word count"])
    print()

# calculate the average number of words per message
def calcAvrgWordsPerMessage(df: pd.DataFrame):
    print("Average number of words per message")
    data = df.groupby("Name") \
        .agg({
            "Message": "count",
            "Message length": "sum",
            "Message word count": "sum"
        })
    data["length/message"] = data["Message length"] / data["Message"]
    data["words/message"] = data["Message word count"] / data["Message"]
    print(data.sort_values(by="words/message", ascending=False))
    print()

# returns the frequency of words a member of the group has said
def getUserWordFrequency(df: pd.DataFrame, name: str = "", top_n: int = 100):
    # filter by name or select messages sent by all
    if name == "":
        userMsg = df
    else:
        userMsg = df[df["Name"] == name]

    # drop message <Medien ausgeschlossen>
    userMsg = userMsg[(userMsg["Message"] == "<Medien ausgeschlossen>") == False]

    # ignore if word is in upper or lowercase
    # remove chained assignment: userMsg["Message"] = userMsg["Message"].str.lower()
    userMsg.loc[:,"Message"] = userMsg.loc[:, "Message"].str.lower()

    # source: https://stackoverflow.com/questions/64022617/counting-occurrences-of-word-in-a-string-in-pandas
    # give every word its own row
    userMsg = userMsg.assign(word = userMsg["Message"].str.split()) \
                            .explode("word")

    # remove special chars before or after words
    userMsg["word"] = userMsg["word"].str.strip(",:().*?!")

    # count all occurrences of words in Justin's messages
    counts = userMsg[["word", "Message word count"]] \
                .groupby("word").count() \
                .sort_values(by="Message word count", ascending=False)

    counts = counts.reset_index()
    print(counts.head(top_n))
    return counts

def getMessageFrequencyPerHour(df: pd.DataFrame, plot: bool = False):
    data = df[["Time", "Name", "Message"]]
    
    # set time of every message to full hour
    hour_value = pd.to_datetime(data["Time"], format="%H:%M")
    hour_value = hour_value.dt.hour
    data.loc[:, "Time"] = hour_value

    # count the messages sent in this hour
    data = data.groupby("Time")["Message"].count().to_frame()
    if plot:
        data.plot(kind="bar", xlabel="Hour", ylabel="Messages sent")
    return data

def getMessageFrequencyPerMemberPerHour(df: pd.DataFrame, plot: bool = False):
    data = df[["Time", "Name", "Message"]]

    # set time of every message to full hour
    hour_value = pd.to_datetime(data["Time"], format="%H:%M")
    hour_value = hour_value.dt.hour
    data.loc[:, "Time"] = hour_value

    # count the messages sent in this hour
    data = data.groupby(["Name", "Time"])["Message"].count().to_frame()
    
    # pivot table for more compact display
    data.reset_index(inplace=True)
    data = pd.pivot_table(data=data, index='Time', columns='Name', values='Message')
    
    # fill NaN with 0 and change datatype back to int
    data = data.fillna(0).astype(int)
    
    if plot:
        data.plot(kind="bar", subplots=True, legend=False, figsize=(10,35), ylabel="Number of messages")
    return data

def getMessageFrequencyPerDay(df: pd.DataFrame):
    msgPerDay = df.groupby("Date")["Date"] \
                .count() \
                .reset_index(name="Number of messages")
    
    # save Date as Datetime object to better sort 
    msgPerDay["Date"] = pd.to_datetime(msgPerDay["Date"], format="%d.%m.%y")
    msgPerDay = msgPerDay.sort_values(by="Date")

    return msgPerDay

def plotAvrgNumberOfMessagesInTimeFrame(df: pd.DataFrame, time_frame: str):
    # per day
    if time_frame == "Day":
        msgPerDay = getMessageFrequencyPerDay(df)
        msgPerDay["Date"] = msgPerDay["Date"].dt.dayofweek
        msgPerDay = msgPerDay.groupby("Date").mean()
        return msgPerDay.plot.bar(xlabel="Weekday", ylabel="Number of messages sent")

    # per week
    if time_frame == "Week":
        msgPerWeek = getMessageFrequencyPerDay(df)
        msgPerWeek["Date"] = msgPerWeek["Date"].dt.isocalendar().week
        msgPerWeek = msgPerWeek.groupby("Date").mean()
        return msgPerWeek.plot.bar(xlabel="Week", ylabel="Number of messages sent", figsize=(20,10))

    # per month
    if time_frame == "Month":
        msgPerMonth = getMessageFrequencyPerDay(df)
        msgPerMonth["Date"] = msgPerMonth["Date"].dt.month
        msgPerMonth = msgPerMonth.groupby("Date").mean()
        return msgPerMonth.plot.bar(xlabel="Month", ylabel="Number of messages sent")

def showUseOfWordsOverTime(df: pd.DataFrame, word: str, time_frame_in_days: int, name: str = ""):
    # use only the messages of a certain person if specified
    if name:
        df = df[df["Name"] == name]
    
    # set the date as the index
    df.loc[:, "Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%y %H:%M")
    df = df.sort_values(by="Datetime")
    df.index = df["Datetime"]

    # filter out the messages where one of the words are mentioned
    word_mentions = df[df["Message"].str.contains(word, na=False)]

    # create rolling window of mentions
    word_rolling = word_mentions["Message"].rolling(str(time_frame_in_days) + "D").count()

    # plot the rolling series
    return word_rolling.plot(kind="line")

# analyses an export of a WhatsApp Chat
def startAnalysis():
    origFilePath = "./[Chatfile].txt"
    preprocessedFilePath = "Chats/Preprocessed" + origFilePath
    origFilePath = "Chats/" + origFilePath

    # processing data
    preprocessFile(origFilePath, preprocessedFilePath)
    df = convertFileToDataframe(preprocessedFilePath)
    df = postprocessData(df)

    # analysing data
    countMessagesByName(df)
    countWordsByName(df)
    calcAvrgWordsPerMessage(df)
    getUserWordFrequency(df)
    getMessageFrequencyPerHour(df, plot=True)
    getMessageFrequencyPerMemberPerHour(df, plot=True)
    plotAvrgNumberOfMessagesInTimeFrame(df, "Day")
    showUseOfWordsOverTime(df, "Word", time_frame_in_days=60)
    plt.show()


if __name__ == "__main__":
    startAnalysis()