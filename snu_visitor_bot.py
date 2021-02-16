from telegram.ext import Updater, MessageHandler, Filters, CommandHandler  # import modules
import os
import pandas as pd

my_token = '1689881196:AAHtrVDGWg-huAYP9NCRzHUDAzIHTVMewqg'
photo_dir = 'C:/Users/hjeon/2021_InternshipSSAI/face_data/'
photo_dir = './face_data/'

photo_name = 0
print('start telegram chat bot')

# message reply function
def get_message(update, context) :
    global photo_name
    message = update.message.text
    if len(message) == 16:
        year, month, day, sh, eh = int(message[:4]), int(message[5:7]), int(message[8:10]), int(message[11:13]), int(message[14:16])
        data = pd.read_csv("pass.csv")
        filter = (data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] > sh) & (data['hour'] <= eh)
        filtered = data[filter]
        if filtered.size:
            t_kp, t_ukp, t_kpa, t_ukpa = filtered['known_ppl'].sum(), filtered['unknown_ppl'].sum(), filtered['known_pass'].sum(), filtered['unknown_pass'].sum()
            update.message.reply_text("Known People :"+str(t_kp)+"\n" + "Unknown People : "+str(t_ukp)+"\n" +
                                      "Pass of known people : "+str(t_kpa)+"\n" + "Pass of unknown people : "+str(t_ukpa))
        else:
            update.message.reply_text("No one passed")
    if photo_name:
        os.rename(photo_dir + 'tmp.png', photo_dir + message + '.png')
        update.message.reply_text("Face registered")
        photo_name = 0
    else:
        update.message.reply_text("You can view the statics when you enter <YYYY.MM.DD.HH-HH> format")



# help reply function
def help_command(update, context) :
    update.message.reply_text("You can view the statics when you enter <YYYY.MM.DD.HH-HH> format")


# photo reply function
def get_photo(update, context) :
    global photo_name
    file_path = photo_dir + 'tmp.png'
    photo_id = update.message.photo[-1].file_id
    photo_file = context.bot.getFile(photo_id)
    photo_file.download(file_path)
    update.message.reply_text('Please enter your name')
    photo_name = 1


updater = Updater(my_token, use_context=True)

message_handler = MessageHandler(Filters.text & (~Filters.command), get_message) # 메세지중에서 command 제외
updater.dispatcher.add_handler(message_handler)

help_handler = CommandHandler('help', help_command)
updater.dispatcher.add_handler(help_handler)

photo_handler = MessageHandler(Filters.photo, get_photo)
updater.dispatcher.add_handler(photo_handler)

updater.start_polling(timeout=3, clean=True)
updater.idle()
