CS6476 Final Project

Tianfang Xie
txie35@gatech.edu

File Description------------------------------------------

There are two folders named input_images and graded_images in the zip file.
	The input_images folder contains five images that are used to generate the final results.
		These files are:
		input1.jpg from https://creativecaincabin.com/wp-content/uploads/2016/04/IMG_0438.jpg
		input2.jpg from https://i.pinimg.com/originals/06/b7/1e/06b71e9be257354fdf93e7ca152ca39e.jpg
		input3.jpg from https://petticoatjunktion.com/wp-content/uploads/2013/11/front-door-house-numbers_thumb.jpg
		input4.jpg from https://images-na.ssl-images-amazon.com/images/I/81U2QeF%2BvuL._AC_SL1500_.jpg
		input5.jpg from https://res.cloudinary.com/woodland/image/upload/c_limit,d_ni.png,f_auto,q_auto,w_1024/v1/craftcuts_media/media/catalog/product/h/o/house-numbers.jpg
		The input5.jpg has been rotated to test the model's prediction of slanted numbers
	The graded_images will store the output data
	
There are four .py files in the zip file.
	The run.py will be used for the grading and output generation.
	The cnn_detection_model.py is used to generate the model to seperate digits and non-digits, and generate the digit_detection_model.h5 file.
	The cnn_classification_model.py is used to generate the model to identify the numbers from 0 to 9 and genegrate the digit_classification_model.h5 file.
	The vgg_pretrain_model.py is used to generate the pre-trained VGG16 model and generate the VGG_pretrain_model.h5 file.
	
There are three h5 files in the zip file.
	The digit_detection_model.h5 file and the digit_classification_model.h5 file will be used in run.py.

The environment setting file is cv_proj.yml.

The zip file has the structure as:
	- zip file:
		- input_images folder:
			- input1.jpg
			- input2.jpg
			- input3.jpg
			- input4.jpg
			- input5.jpg
		- graded_images folder
		- run.py
		- cnn_detection_model.py
		- cnn_classification_model.py
		- vgg_pretrain_model.py
		- digit_detection_model.h5
		- digit_classification_model.h5
		- VGG_pretrain_model.h5
		- cv_proj.yml
		- README.md

How to run the code------------------------------------------

	1. Make sure you have the correct cv_proj.yml that I provided.
	2. Make sure there are five input images in the input_images folder.
	3. Make sure you have the digit_detection_model.h5 file and the digit_classification_model.h5 file
	3. Now install the environment described in cv_proj.yml using the following command:
	   conda env create -f cv_proj.yml
	4. Activate the environment run the command:
	   conda activate cv_proj
	5. Now you can use python to run the run.py file and check the output images in the graded_images folder.