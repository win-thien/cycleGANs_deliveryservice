# cycleGANs_deliveryservice
Implementing a cycleGAN to produce generated images in the style of Studio Ghibli. 

**Contact:**
https://github.com/win-thien/cycleGANs_deliveryservice
https://www.linkedin.com/in/thien-win-pe/


**Project Name:** CycleGANs Delivery Service


**Description:**

Style Transfer is a form of translation where the goal is to map the features of a source subject (Domain A) onto another receiver subject (Domain B). This can be broadened to still images, video content or even audio. The focus of this project was to map the learned features of Studio Ghibli style animation to real life photos utilizing a cycle generative adversarial network (cycleGAN). A main difference in cycleGANs versus other style transfer techniques is the use of unpaired images.

Studio Ghibli is a famous animation production company from Japan and are the masterminds behind great animated movies such as Spirited Away, My Neighbor Totoro and Grave of the Fireflies. Their movies carry distinct, thought provoking messages filtered through the lens of animation. Kiki’s Delivery Service for example, for which this repository name pays homage to, teaches us about work-life balance, burnout, and that personal and professional growth is at times scary.

My hope is you enjoy this project as much as I enjoyed working on it to get to this point. Have fun creating!


**Business Case:**

This project was strictly for pleasure and for my capstone at BrainStation but the theory and methods could be broadened to a business use case. The social media filters employed often employ a form of neural style transfer where a video or image is converted to look like something else (i.e. Aging Filter, Disney Character Filter, etc.)   This adds value to the platform by potentially increasing user engagement. 


**Associated Links:**
Google Drive(Available until 4/30/2022)
GitHub Repository


**Associated Files:**
In this repository you will find the following files created for this project:
* 01 Data Scraping and Wrangling
* 02 CycleGAN Training
* 03 Model Evaluation
* 04 FID Score
* cycleGAN_functions.py
* cycleGAN_model.py 
* Training_VerboseOutput.csv
* training_ckpts

Notebooks are meant to be worked in order from 01 to 04 with the other files being used to supplement these notebooks. 


**Installation:**
The repository contains a .yml file that contains all the required libraries used to create a virtual environment that can be imported into Anaconda.
* cycleGAN_SG_env.yml 


**Acknowledgements and Resources:**
I would like to take this section to personally thank the Education Team at BrainStation for winter class 2022. Your guidance in my data science journey and education has been paramount in executing this project.

This project also wouldn’t have been possible if not for the multitude of available resources from the online community. This project was originally inspired by a Kaggle competition to convert photographs to Monet style paintings. The following links can help expand your understanding of cycleGAN’s and neural style transfer in general.
* https://arxiv.org/pdf/1910.00927.pdf 
* https://neptune.ai/blog/6-gan-architectures
* https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial
* https://www.youtube.com/watch?v=2MSGnkir9ew&ab_channel=DigitalSreeni 
* https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
* https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ 
 


**Roadmap:**
With continued iterations, I will be looking to complete and implement the following:
* ResNet based generator
* Employing methods to prevent mode collapse
* Address non-converging model
* Employ a different dataset with more images



**Project Status: Ongoing**
This project was submitted for the purposes of fulfilling my BrainStation Capstone project requirement. Continued development will occur and improvements are sought after. I am open to partnering with the community to improve output fidelity.
