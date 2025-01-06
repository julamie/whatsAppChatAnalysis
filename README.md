# Show statistics from your WhatsApp chats
You can use this little script to see some useful statistics about your private or group chat, e.g. how many messages someone sent.

# Requirements
* Matplotlib (https://matplotlib.org/stable/install/index.html)
* NumPy (https://numpy.org/install/)
* Pandas (https://pandas.pydata.org/getting_started.html)
* Seaborn (https://seaborn.pydata.org/installing.html)  

# How do I run this?
* Export your WhatsApp chat of your choice (https://faq.whatsapp.com/1180414079177245)
* Change the variable origFilePath in startAnalysis() to the path where you saved the exported chat file
* Run the script, comment out any functions in startAnalysis() you don't want to run

# What statistics can be shown?
* Show how many messages or words a person has sent
* Show the average message length of a person
* Show the top 100 words of somebody
* Show the number of sent messages, grouped by the hour of the day, or day
* Show the average number of messages sent over a day, week or month
* Show the number of messages per week, shown as a line for every person
