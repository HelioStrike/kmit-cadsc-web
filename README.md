# kmit-cadsc-web

## WATCH OUT. WORK IN PROGRESS!!!

A showcase of the work (being) done in the KMIT CADSC project.

To view the website, simply run `python3 server.py` and then go to `http://localhost:5010`.

# For colleagues working on the project


## Adding your task to the website

Let us say you need to add a new task to the website, called "Pizza Grading".

### Creating a webpage for your task

- Go to `templates/partials/header.html` and inspecting the line above "Add new task button here", add a similar line for your task.
For Pizza Grading, this line would look like:
```
<a class="item" href="/tasks/pizza-grading">Pizza Grading</a>
```
- Create a new HTML file in `templates` named as your task in snake case ("pizza_grading.html").
- Copy the contents of `templates/epithelium_segmentation.html`, paste it into your newly created HTML file and edit it as you see fit. The parts that you might want to edit are highlighted with "Might want to edit this".
- Now open up the main page, click on Menu, and you should be able to access your new page!

### Writing the image processing code for the 'Test it out!' section

- Create a new folder named as your task in snake case ("Pizza Grading" => "pizza_grading").
- Here, create a file `__init__.py`. This is where you write all the image processing code. Write a function that takes in a file name, and generates the output image.
- When the 'Go!' button is pressed, the function `process_image` in `server.py` is called, so you may need to edit `process_image` to work with the above code.
- If everything goes well, taking in input images in the form and hitting 'Go!' should redirect you to the output image.

### Finishing touches

- Add sample inputs of your task to the `sample_inputs` directory.

### Updating the repo

- Just create a pull request against master.