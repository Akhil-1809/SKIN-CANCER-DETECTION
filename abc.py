import tempfile
from tracemalloc import start
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import cv2
from main import *
def gui():
	def set_bg_hack_url():
		'''
		A function to unpack an image from url and set as bg.
		Returns
		-------
		The background.
		'''
			
		st.markdown(
			f"""
			<style>
			.stApp {{
				background: url("https://cdn.pixabay.com/photo/2020/06/19/22/33/wormhole-5319067_960_720.jpg");
				background-size: cover
			}}
			</style>
			""",
			unsafe_allow_html=True
		)
	
		
	st.title('SKIN CANCER')
	st.sidebar.title('Parametres for detection')
	st.markdown(
		'''
		<style>
		[data-testid='sidebar'][aria-expanded='true'] > div:firstchild{width:400px}
		[data-testid='sidebar'][aria-expanded='false'] > div:firstchild{width:400px , margin-left: -400px}
		</style>
		''',
		unsafe_allow_html=True
	)

	st.sidebar.markdown('---')
	st.sidebar.title("Image source")

	source = st.sidebar.file_uploader('Upload Image ' , type=['jpg','png','jpeg'])
	demo_image = r'"C:\Users\chait\OneDrive\Desktop\AKHIL\download (9).jpg"'
	# st.image(demo_image, caption='Image of a Fabric')
	image = cv2.imread(demo_image)
	st.image(image, caption='Image of a skin')
	st.write("""Skin cancer 
	the abnormal growth of skin cells 
	most often develops on skin exposed to the sun. But this common form of cancer can also occur on areas of your skin not ordinarily exposed to sunlight. There are three major types of skin cancer — basal cell carcinoma, squamous cell carcinoma and melanoma.""")

	
	tfile = tempfile.NamedTemporaryFile(suffix = '.png', delete=False) 


	# Checking if the file is being run as a script or imported as a module.
	if not source:
		vid = cv2.imread(demo_image)
		tfile.name = demo_image
		dem_img = open(tfile.name , 'rb')
		demo_bytes = dem_img.read()

		st.sidebar.text("Input Image")
		st.sidebar.image(demo_bytes)       
		
		# st.video(demo_bytes) #displaying the video
	else:
		tfile.write(source.read())
		dem_vid = open(tfile.name , 'rb')
		demo_bytes = dem_vid.read()

		st.sidebar.text("Input Image")
		st.sidebar.image(demo_bytes)

		# st.video(demo_bytes) #displaying the video

			
	print(tfile.name)
	
	
	Analysis = st.sidebar.button('Analysis')
	
	Start = st.sidebar.button('Start')

	stop = st.sidebar.button('Stop')

	if Analysis:
		ab=analysis()
		st.text(ab)

	if Start: 
		ab = run(source = tfile.name)
		st.text(ab)
   
	if stop:
		# vid.release()
		Start = False
		print(Start)
		st.text("Processing has ended, you may close the tab now.")
	

if __name__ == '__main__':
	try:
		gui()
	except SystemExit:
		pass
