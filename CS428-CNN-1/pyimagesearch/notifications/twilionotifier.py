# import the necessary packages
from twilio.rest import Client
from datetime import datetime
from datetime import date

class TwilioNotifier:
	def __init__(self, conf):
		# store the configuration object
		self.conf = conf

	def send(self, msg):
		# initialize the twilio client and send the message
		client = Client(self.conf["twilio_sid"],
			self.conf["twilio_auth"])
		client.messages.create(to=self.conf["twilio_to"], 
			from_=self.conf["twilio_from"], body=msg)