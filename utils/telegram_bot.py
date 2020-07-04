#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements a telegram bot to delivery some pieces of information about the training phase.

If you find any bug or have some suggestion, please, email me.
"""
from datetime import date

from telegram.ext import Updater, CommandHandler
import datetime

class TelegramBot:
    """
    Using this class you're going to be able to send some messages via telegram. You may, for example, get to know
    when a training is over and what's the final stats about it.

    To know more about how the telegram bot works go to: https://github.com/python-telegram-bot/python-telegram-bot
    """

    def __init__(self, chat_id, token="821262177:AAFwwfIc7tkJuwYipyD89hGyF-qyJmeX6a4", model_name="CNN"):
        """
        Class contructor
        :param chat_id (string): the id in which the bot needs to send a message
        :param token (string, optional): the bot token. The default is the Jedy-Bot
        :param model_name (string, optional): the model's name, ex: ResNet. Default is CNN
        """
        self.token = token
        self.chat_id = chat_id
        self.model_name = model_name
        self.info = False
        self.epoch_info = "Hey, it's running the 1st epoch yet!"
        self.best_info = "Calm down! Wait to finish the 1st epoch to get the best performance so far."

    def start_bot (self):
        """
        This method just start the bot and send a msg saying it's about to start. The user can interact with the bot
        sendind /info, /stop and /best. This commands will be get by the CommandHandler and will change the class'
        attributes. In this sense, the training loop will check if it needs to send any information to through the bot.
        """
        self.updater = Updater(token=self.token)
        self.updater.start_polling()

        # Setting a dispatcher to interact via app
        disp = self.updater.dispatcher

        info_handler = CommandHandler("info", self.get_info)
        disp.add_handler(info_handler)

        stop_handler = CommandHandler("stop", self.stop_info)
        disp.add_handler(stop_handler)

        best_handler = CommandHandler("best", self.get_best_info)
        disp.add_handler(best_handler)

        epoch_handler = CommandHandler("epoch", self.get_epoch_info)
        disp.add_handler(epoch_handler)

        good_handler = CommandHandler("goodbot", self.get_good_bot)
        disp.add_handler(good_handler)

        now = datetime.datetime.now().strftime("%d/%m/%Y -- %H:%M")


        self.updater.bot.send_message(chat_id=self.chat_id,
                                      text="--------\nHello, the training phase of your {} model is about to start!\nDate and time: {}\n\nSend /info to check the status every epoch. By default, I won't send it except you ask.\n\nSend /stop to stop to check the status.\n\nSend /best to get the best performance so far.\n\nSend /epoch to get the current epoch so far.\n--------\n".format(self.model_name, now))

    def send_msg (self, msg):
        self.updater.bot.send_message(chat_id=self.chat_id, text=msg)

    def get_info (self, update, context):
        self.info = True

    def stop_info (self, update, context):
        self.info = False

    def get_best_info (self, update, context):
        self.send_msg(self.best_info)
        
    def get_epoch_info (self, update, context):
        self.send_msg(self.epoch_info)

    def get_good_bot (self, update, context):
        self.send_msg("Uhuuuul! Now can you pay me a coffee?")

    def stop_bot (self):
        self.updater.stop()




