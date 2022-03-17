# cse160-WI22
Materials for CSE 160 Programming Assignments

## Access DSMLP using ssh

You will use the [UCSD Data Science / Machine Learning Platform (DSMLP)](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=fda9846287908954947a0fa8cebb352b) to get access to a GPU. 

You can login to DSMLP using by `ssh USERNAME@dsmlp-login.ucsd.edu`. Your username and password are your UCSD account. You can set up an [ssh key](https://support.ucsd.edu/services?id=kb_article_view&sys_kb_id=711d8e9e1b7b34d473462fc4604bcb47) that allows you to more easily login. 

DSMLP uses containers to set up its software environment. You must create a container that provides access to a GPU with CUDA installed using the command ` launch.sh -g 1 -s -i ucsdets/nvcr-cuda:latest`

Once you have that container, you can compile and run the Makefiles in the PA directories.

Please be considerate on your use of the GPUs. The GPUs in DSMLP are shared within this class and across campus. If you are not actively using the GPU, you should shut down the container to allow others to access it.


## Access DSMLP using VSCode

It is possible to access DSMLP using a local version of VSCode. 

Steps:

1. Install VS Code https://code.visualstudio.com/download
2. Install Remote-SSH plugin by searching for it in the extensions view
3. Click on the indicator on the bottom left corner

![image](https://user-images.githubusercontent.com/43923184/148268541-202b9806-7d08-415b-ad4d-7b4d04916388.png)

4. Click on Connect to Host.. and + Add new SSH Host...
5. Type in USERNAME@dsmlp-login.ucsd.edu (USERNAME to be replaced with your UCSD Active Directory username)
6. Click on where you want to save the SSH Configuration
7. Click on the Connect Popup
8. Type your UCSD password when prompted and press enter
9. You are now connected to UCSD DSMLP! It can be verified by checking the bottom left corner which indicates dsmlp-login.ucsd.edu

A video is attached in case there are any issues with following the steps:

https://user-images.githubusercontent.com/43923184/148276847-f92fdbd4-14a4-4749-9b89-615c64b7ad81.mp4

## Access to CUDA and GPU:

Open terminal on VSCode when connected to DSMLP and run `launch.sh -g 1 -s -i ucsdets/nvcr-cuda:latest`
You should see an output like this:

![image](https://user-images.githubusercontent.com/43923184/148271105-200ed36c-dc88-4b01-9b68-cdb61a36b655.png)

This gives you access to GPU infrastructure on DSMLP; it starts a container with GPU access and loads it with a software image that contains CUDA and other basic packages. 

You must be within GPU container in order to properly compile. If you get an error about not having access to nvcc, then you are not in the container.

Please only use the container when you are compiling and release it when you are completed. 
