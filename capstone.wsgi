#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.append("/var/www/Capstone_DL_Object_detection/")

from app import app as application
application.secret_key = 'Add your secret key'

