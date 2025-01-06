import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# opens file, clears it from weird characters and saves result in new file
def preprocessFile(oldPath: str, newPath: str):
    with open(oldPath, encoding="utf-8") as origFile:
        data = origFile.read()

    data = removeZeroWidthSpace(data)
    data = escapeDoubleQuotesInData(data)
    data = convertTextToCSVFormat(data)
    
    with open(newPath, 'w', encoding="utf-8") as newFile:
        newFile.write(data)

# replace zero width character with empty character
def removeZeroWidthSpace(data: str):
    return data.replace(u'\u200b', "")

# escape double quotes with \ to prevent messing with the data when converting
def escapeDoubleQuotesInData(data: str):
    data = data.replace('"', r'\"')
    return data

# edit format of file to csv, add quotation marks around messages and end message with EXT (End of text)
def convertTextToCSVFormat(data: str):
    # Line format: Date, Time - Sender: Message
    # Allowed date format examples 24.12.24, 24.12.2024, 24/12/24, 24/12/2024
    pattern = re.compile(r"(?P<Timestamp>\d{2}[./]\d{2}[./](\d{2}|\d{4}), \d{2}:\d{2}) - (?P<Message>.*)")
    timestamp, name, message = "Timestamp", "Name", "Message"

    convertedData = []
    for line in data.split("\n"):
        newMessageRegex = pattern.match(line)

        # determine the parts of the message and save them to converted data as soon as a new message appears
        if newMessageRegex:
            convertedData.append(f"{timestamp}\t{name}\t{message}\x03") # \x03 is the escape char for End of Text
            timestamp = newMessageRegex.group("Timestamp")
            information = newMessageRegex.group("Message")

            (name, message) = splitInformation(information)
        # the message continues at a new line, add them to the current message
        else: 
            message += "\n" + line

    return "\n".join(convertedData)

def splitInformation(information: str):
    # sender name is either the contact name or a telephone number
    informationRegex = re.match(r"(?P<Sender>([\w\s]+|([\+\d\s]+))): (?P<Text>.*)", information)

    # if there is no sender, the message has been an information from the WhatsApp application
    # example: X has left the chat
    if not informationRegex:
        return ("WhatsApp", '"' + information + '"')

    # surround message by double quotes
    return (informationRegex.group("Sender"), '"' + informationRegex.group("Text") + '"')

def convertFileToDataframe(path: str):
    df = pd.read_csv(path, usecols=range(3), skipinitialspace=True,
                     lineterminator="\x03", escapechar='\\',
                     delimiter="\t", parse_dates=["Timestamp"],
                     date_format="mixed", dayfirst=True, 
                     encoding="utf-8")
    # show full length of message when printing
    pd.set_option("max_colwidth", None)
    pd.set_option("max_seq_item", None)

    return df

# cleans the dataframe after creation
def postprocessData(df: pd.DataFrame):
    # add new data
    addColumnMessageLength(df)
    addColumnNumberOfWords(df)

    # replace false data
    replaceNanMessages(df)
    return df

# adds length of message as a column to dataframe
def addColumnMessageLength(df: pd.DataFrame):
    df["Message length"] = df["Message"].str.len()

# adds number of words in message as a column
def addColumnNumberOfWords(df: pd.DataFrame):
    df["Message word count"] = df["Message"].apply(lambda x: len(str(x).split(' ')))

# changes messages which are NaN to the correct ones: One time messages
def replaceNanMessages(df: pd.DataFrame):
    df["Message"] = df["Message"].fillna("<Media omitted>")
    df["Message length"] = df["Message length"].fillna(15)
    df["Message word count"] = df["Message word count"].fillna(2)

# count how many messages everyone sent
def countMessagesByName(df: pd.DataFrame):
    return df.groupby("Name")["Message"] \
             .count() \
             .sort_values(ascending=False)

# count how many words everyone sent
def countWordsByName(df: pd.DataFrame):
    return df.groupby("Name")["Message word count"] \
             .sum() \
             .sort_values(ascending=False)

# calculate the average number of words per message
def calcAvrgWordsPerMessage(df: pd.DataFrame):
    data = df.groupby("Name") \
        .agg({
            "Message": "count",
            "Message length": "sum",
            "Message word count": "sum"
        })
    
    data["length/message"] = data["Message length"] / data["Message"]
    data["words/message"] = data["Message word count"] / data["Message"]
    return data.sort_values(by="Message", ascending=False)

# returns the frequency of words a member of the group has said
def getUserWordFrequency(df: pd.DataFrame, name: str = "", top_n: int = 100):
    # filter by name or select messages sent by all
    if name == "":
        userMsg = df
    else:
        userMsg = df[df["Name"] == name]

    # drop message <Media omitted> and <Medien ausgeschlossen> (german)
    userMsg = userMsg[(userMsg["Message"] == "<Media omitted>") == False]
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

    # count all occurrences of words in the messages
    counts = userMsg[["word", "Message word count"]] \
                .groupby("word").count() \
                .sort_values(by="Message word count", ascending=False)

    counts = counts.reset_index()

    return counts.head(top_n)

def getMessageFrequencyPerHour(df: pd.DataFrame, plot: bool = False):
    # set time of every message to full hour
    df["Hour"] = df["Timestamp"].dt.hour

    # count the messages sent in this hour
    df = df.groupby("Hour")["Message"].count().to_frame()
    if plot:
        df.plot(kind="bar", title="Number of messages per hour", xlabel="Hour", ylabel="Messages sent")
    return df

def getMessageFrequencyPerMemberPerHour(df: pd.DataFrame, plot: bool = False):
    # set time of every message to full hour
    df["Hour"] = df["Timestamp"].dt.hour

    # count the messages sent in this hour
    df = df.groupby(["Name", "Hour"])["Message"].count().to_frame()
    
    # pivot table for more compact display
    df = pd.pivot_table(data=df, index='Hour', columns='Name', values='Message')
    
    # fill NaN with 0 and change datatype back to int
    df = df.fillna(0).astype(int)
    
    if plot:
        numberOfUsers = df.shape[1]
        df.plot(kind="bar", subplots=True, 
                legend=False, figsize=(10, 3 * numberOfUsers), ylabel="Number of messages")
    return df

def getMessageFrequencyPerDay(df: pd.DataFrame):
    df["Date"] = df["Timestamp"].dt.date
    msgPerDay = df.groupby("Date")["Date"] \
                .count() \
                .reset_index(name="Number of messages")
    
    # save Date as Datetime object to better sort 
    msgPerDay["Date"] = pd.to_datetime(msgPerDay["Date"])
    msgPerDay = msgPerDay.sort_values(by="Date")

    return msgPerDay

def plotAvrgNumberOfMessagesInTimeFrame(df: pd.DataFrame, time_frame: str):
    # per day
    if time_frame == "Day":
        msgPerDay = getMessageFrequencyPerDay(df)
        msgPerDay["Date"] = msgPerDay["Date"].dt.dayofweek
        msgPerDay = msgPerDay.groupby("Date").mean()

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        plot = msgPerDay.plot.bar(xlabel="Weekday", title = "Average number of messages per day of the week",
                                  ylabel="Number of messages sent")
        plot.set_xticklabels(day_names)
        return plot

    # per week
    if time_frame == "Week":
        msgPerWeek = getMessageFrequencyPerDay(df)
        msgPerWeek["Date"] = msgPerWeek["Date"].dt.isocalendar().week
        msgPerWeek = msgPerWeek.groupby("Date").mean()

        return msgPerWeek.plot.bar(xlabel="Week", title= "Average number of messages for every week of the year",
                                   ylabel="Number of messages sent", figsize=(20,10))

    # per month
    if time_frame == "Month":
        msgPerMonth = getMessageFrequencyPerDay(df)
        msgPerMonth["Date"] = msgPerMonth["Date"].dt.month
        msgPerMonth = msgPerMonth.groupby("Date").mean()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plot = msgPerMonth.plot.bar(xlabel="Month", title= "Average number of messages for every month of the year",
                                    ylabel="Number of messages sent")
        plot.set_xticklabels(month_names)
        return plot

def showMessageCountOverTime(df: pd.DataFrame, names: list = []):
    # use only the messages of certain persons if specified
    if names:
        data = df[df["Name"].isin(names)]
    else:
        # ignore meta data (pun unintended)
        data = df[df["Name"] != "WhatsApp"]

    # save the week of sent messages
    data.loc[:, "Week"] = data.loc[:, "Timestamp"].dt.to_period("W")

    # count the number of messages per week per person
    data = data.groupby(["Name", "Week"])["Message"].count().reset_index()
    data = data.pivot(index="Week", columns="Name", values="Message")

    return data.plot(kind="line", legend=True)

# analyses an export of a WhatsApp Chat
def startAnalysis():
    origFilePath = "[CHAT_EXPORT_FILE].txt"
    preprocessedFilePath = "Chats/Preprocessed" + origFilePath
    origFilePath = "Chats/" + origFilePath

    # processing data
    preprocessFile(origFilePath, preprocessedFilePath)
    df = convertFileToDataframe(preprocessedFilePath)
    df = postprocessData(df)

    # analysing data
    print("Sent messages by user")
    print(countMessagesByName(df))

    print("Sent words by user")
    print(countWordsByName(df))
    
    print("Average number of words per message")
    print(calcAvrgWordsPerMessage(df))

    print("Most used words in chat")
    print(getUserWordFrequency(df)) # optional: name= and top_n=100

    getMessageFrequencyPerHour(df, plot=True)

    getMessageFrequencyPerMemberPerHour(df, plot=True)

    plotAvrgNumberOfMessagesInTimeFrame(df, time_frame="Day") # time frames: "Day", "Week", "Month"

    showMessageCountOverTime(df, "[INSERT_WORD]") # names= optional
    plt.show()


if __name__ == "__main__":
    startAnalysis()