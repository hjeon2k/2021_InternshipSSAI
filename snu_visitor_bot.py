from telegram.ext import Updater, MessageHandler, Filters, CommandHandler  # import modules
import os
import pandas as pd
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math

my_token = '*********************************************'
photo_dir = 'C:/Users/hjeon/2021_InternshipSSAI/face_data/'
photo_dir = './face_data/'
photo_name = 0



# message reply function
def get_message(update, context):
    global photo_name
    message = update.message.text
    if len(message.split('-'))==2 and len(message.split('-')[0].split('.'))==4 and len(message.split('-')[1].split('.'))==4:
        begin = datetime.strptime(message.split('-')[0], '%Y.%m.%d.%H')
        end = datetime.strptime(message.split('-')[1], '%Y.%m.%d.%H')
        data = pd.read_csv("pass.csv")
        data_filter = lambda data : (datetime(data['year'], data['month'], data['day'], data['hour']) >= begin) & (datetime(data['year'], data['month'], data['day'], data['hour']) < end)
        filtered = data[data.apply(data_filter, axis=1)]
        if filtered.size:
            t_kp, t_ukp, t_kpa, t_ukpa = filtered['known_ppl'].sum(), filtered['unknown_ppl'].sum(), filtered['known_pass'].sum(), filtered['unknown_pass'].sum()
            update.message.reply_text("Total Visitors :"+str(t_kp + t_ukp)+"\n" + "Total number of Visits : "+str(t_kpa + t_ukpa))
            fig, ax = plt.subplots()
            cmap = plt.get_cmap("tab20c")
            pass_labels, pass_chart, pass_colors = ['Total visit by\nRegistered People', 'Total visit by\nStrangers'], np.array([t_kpa, t_ukpa]), cmap(np.array([1, 5]))
            visitor_labels, visitor_chart, visitor_colors = ['Registered\nPeople', 'Strangers'], np.array([t_kp, t_ukp]), cmap(np.array([2, 6]))
            pass_value = lambda val: '{:.0f}'.format(np.round(val / 100 * pass_chart.sum()))
            visitor_value = lambda val: '{:.0f}'.format(np.round(val / 100 * visitor_chart.sum()))
            ax.pie(pass_chart, labels=pass_labels, autopct=pass_value, radius=1, colors=pass_colors, wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2), pctdistance=0.85, labeldistance=1.1)
            ax.pie(visitor_chart, labels=visitor_labels, autopct=visitor_value, radius=0.65, colors=visitor_colors, wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2), pctdistance=0.75, labeldistance=0.15)
            ax.set(aspect="equal", title='Number of visit and visitors')
            plt.savefig('tmp_plot.png', dpi=300)
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('tmp_plot.png', 'rb'))
            os.remove('tmp_plot.png')

        else:
            update.message.reply_text("No one passed")
    if photo_name:
        os.rename(photo_dir + 'tmp_people.png', photo_dir + message + '.png')
        update.message.reply_text("Face registered as " + message + "\nIt will be applied by tomorrow")
        photo_name = 0
    else:
        update.message.reply_text("You can view the statics when you enter\nYYYY.MM.DD.HH-YYYY.MM.DD.HH\nPlease match the format above")



# help reply function
def help_command(update, context):
    update.message.reply_text("You can view the statics when you enter\bYYYY.MM.DD.HH-YYYY.MM.DD.HH\nPlease match the format above")

def help_command(update, context):
    data = pd.read_csv("pass.csv")
    now = datetime.now()
    wd, wk = np.zeros((4, 7)), []
    for i in range(7):
        tdate = now - timedelta(days=6-i)
        wk.append('{year}.{month}.{day}'.format(year=tdate.year, month=tdate.month, day=tdate.day))
        data_filter = lambda data: date(year=data['year'], month=data['month'], day=data['day']) == date(year=tdate.year, month=tdate.month, day=tdate.day)
        filtered = data[data.apply(data_filter, axis=1)]
        if filtered.size:
            wd[0][i], wd[1][i], wd[2][i], wd[3][i]= filtered['known_ppl'].sum(), filtered['unknown_ppl'].sum(), filtered['known_pass'].sum(), filtered['unknown_pass'].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle('Weekly Visit Statics')
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.array([2, 5, 1, 5]))
    wd_kp, = ax1.plot(wk, wd[0], marker='.', label='Registered people', color=colors[0])
    wd_ukp, = ax1.plot(wk, wd[1], marker='.', label='Strangers', color=colors[1])
    wd_pt, = ax1.plot(wk, wd[0]+wd[1], marker='.', label='Total visitors', color='grey')
    wd_kpa, = ax2.plot(wk, wd[2], marker='s', markersize=3, linestyle='--', label='Total visit by\nregistered people', color=colors[2])
    wd_ukpa, = ax2.plot(wk, wd[3], marker='s', markersize=3, linestyle='--', label='Total visit by\nstrangers', color=colors[3])
    wd_pat, = ax2.plot(wk, wd[2]+wd[3], marker='s', markersize=3, linestyle='--', label='Total visits', color='grey')
    ax1.legend(handles=[wd_kp, wd_ukp, wd_pt], loc=1)
    ax1.tick_params(labelrotation=30)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(handles=[wd_kpa, wd_ukpa, wd_pat], loc=1)
    ax2.tick_params(labelrotation=30)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('tmp_plot.png', dpi=300)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('tmp_plot.png', 'rb'))
    os.remove('tmp_plot.png')

# photo reply function
def get_photo(update, context):
    global photo_name
    file_path = photo_dir + 'tmp_people.png'
    photo_id = update.message.photo[-1].file_id
    photo_file = context.bot.getFile(photo_id)
    photo_file.download(file_path)
    update.message.reply_text('Please enter your name')
    photo_name = 1

def main_bot():
    print('start telegram chat bot')
    updater = Updater(my_token, use_context=True)

    message_handler = MessageHandler(Filters.text & (~Filters.command), get_message) # 메세지중에서 command 제외
    updater.dispatcher.add_handler(message_handler)

    help_handler = CommandHandler('help', help_command)
    updater.dispatcher.add_handler(help_handler)

    weekly_handler = CommandHandler('weekly', help_command)
    updater.dispatcher.add_handler(weekly_handler)

    photo_handler = MessageHandler(Filters.photo, get_photo)
    updater.dispatcher.add_handler(photo_handler)

    updater.start_polling(timeout=3, clean=True)
    updater.idle()

#main_bot()
