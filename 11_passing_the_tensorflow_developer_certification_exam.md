# Preparing for the TensorFlow Developer Certification

After going through the Zero to Mastery TensorFlow for Deep Learning course, you might be interested in taking the TensorFlow Developer Certification exam.

If so, these steps will help you.

* 📖 **Resource:** [Get the slides](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/slides/11_passing_the_tensorflow_developer_certification_exam.pdf) for this section

## Preface

I took and passed the TensorFlow Developer Certification exam myself shortly after it came out. After which, I wrote an article and made a YouTube video on how I did it and how you can too.

Many of the course materials as well as the document you're reading now, were built with the following two resources in mind:

- 📄 **Read:** [How I got TensorFlow Developer Certified (and how you can too)](https://www.mrdbourke.com/how-i-got-tensorflow-developer-certified/)
- 🎥 **Watch:** [How I passed the TensorFlow Developer Certification exam (and how you can too)](https://youtu.be/ya5NwvKafDk)

![Daniel Bourke's TensorFlow Developer certification](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/11-daniel-bourke-tensorflow-developer-certification.png)
*My [TensorFlow Developer Certificate](https://www.credential.net/3d6eec98-5910-4546-b907-172a7d507de6#gs.53cop5) (delivered after passing the exam).*

## What is the TensorFlow Developer Certification?

[The TensorFlow Developer Certification](https://www.tensorflow.org/certificate), as you might’ve guessed, is a way to showcase your ability to use TensorFlow.

More specifically, your ability to use TensorFlow (the Python version) to build deep learning models for a range of tasks such as regression, computer vision classification (finding patterns in images), natural language processing (finding patterns in text) and time series forecasting (predicting future trends given a range of past events).

## Why the TensorFlow Developer Certification?

My first reason was fun. I wanted to give myself a little challenge to work towards and a reason to read a new book I’d purchased (Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow 2.0).

But two other valid reasons are:

1. To acquire the foundational skills required to build machine learning powered applications (you've already done this with the rest of the course)
2. Showcasing your skill competency to a future employer*
3. Get added to the [TensorFlow Certificate Network](https://developers.google.com/certification/directory/tensorflow)

![TensorFlow Certificate Network](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/11-daniel-bourke-tensorflow-developer-certification-network.png)
*Once you pass the TensorFlow Developer Certification, you get added to the TensorFlow Certificate Network, a resource people from around the world can use to find TensorFlow Certified Developers.*

*You can do this in a multitude of ways. For example, creating a GitHub repo where you share your code projects and a blog where you write about the things you've learned/worked on.

### Certificates are *nice* to have not *need* to have

I got asked whether the certification is necessary in a [livestream Q&A about this course](https://youtu.be/rqAqcFcfeK8).

It’s not.

In the tech field, no certification is *needed*. If you have skills and demonstrate those skills (through a blog, projects of your own, a nice-looking GitHub), that is a certification in itself.

A certification is only one form of proof of skill.

Rather than an official certification, I’m a big fan of starting the job before you have it.

For example, if you have some ideal role you’d like to work for a company as. Say, a machine learning engineer. Use your research skills to figure out what a machine learning engineer would do day to day and then start doing those things.

Use courses and certifications as foundational knowledge then use your own projects to build specific knowledge (knowledge that can’t be taught).

![do certificates guarantee a job tweet](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/11-do-certificates-guarantee-a-job-tweet.png)

*Do certificates guarantee a job? [Tweet by Daniel Bourke](https://twitter.com/mrdbourke/status/1385143193918840835?s=20).*

With that being said if you did want to go for the certification, how would you do it?

## How to prepare (your brain) for the TensorFlow Developer Certification

First and foremost, you should have experience writing plenty of TensorFlow code.

After all, since certification is proof of skill, there's no point in going for certification if you don't have some sort of skill at using TensorFlow (and deep learning in general).

If you've gone through the Zero to Mastery TensorFlow for Deep Learning course, coded along with the videos, done the exercises, you've got plenty of skill to take on the exam.

To prepare, go through the [TensorFlow Developer Certificate Candidate Handbook](https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook.pdf). Use this as your ground truth for the exam.

It's a well-written document so I'm not going to repeat anything from within it here, rather suggest some actions which you might want to take.

### The Skills Checklist

Going through the handbook, you'll come across a section named the Skills checklist. It is what it says it is.

Of course, I'm not going to tell what's actually on the exam. But looking at the skills checklist, you'll find sections on:

1. TensorFlow Developer Skills (TensorFlow fundamentals)
2. Building and training neural networks using TensorFlow 2.x
3. Image classification
4. Natural language processing (NLP)
5. Time series, sequences and predictions

If there are five sections, can you guess how many questions will be on the exam?

Each of the questions requires you to submit a trained model in `.h5` format.

If you've been through the course materials, you know how to do this.

The model you submit is graded on how well it performs. Don't overthink this. If you build a fairly well-performing model, chances are it'll pass. If you think your model needs to improve its performance, go through the steps covered in the [Improving a model section](https://dev.mrdbourke.com/tensorflow-deep-learning/02_neural_network_classification_in_tensorflow/#improving-a-model) of the course. 

You've got 5-hours during the exam to build and train models on the datasets provided. The models will not take long to train (even on CPU). This is plenty of time.

🛠 **Exercises**

1. Read through the TensorFlow Developer Certification Candidate Handbook.
2. Go through the Skills checklist section of the TensorFlow Developer Certification Candidate Handbook and create a notebook which covers all of the skills required, write code for each of these (this notebook can be used as a point of reference during the exam).

![mapping the TensorFlow Developer handbook to code in a notebook](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/11-map-the-skills-checklist-to-a-notebook.png)
*Example of mapping the Skills checklist section of the TensorFlow Developer Certification Candidate handbook to a notebook.*

## How to prepare (your computer) for the TensorFlow Developer Certification

Got TensorFlow skills? 

Been through the TensorFlow Developer Certification Candidate Handbook? 

Decided you're going to take on the exam?

Now it's time to set your computer up.

The exam takes place in PyCharm (a Python integrated developer environment or IDE). If you've never used PyCharm before, not to worry, you can get started using the [PyCharm quick start tutorial](https://www.jetbrains.com/pycharm/learn/). Plus, being able to get setup in a new development environment is part of being a skilled developer.

But wait, PyCharm has a lot going on, how do I know to set it up for the TensorFlow Developer Certification Exam?

There's a guide for that!

The TensorFlow team have written a guide (similar to the handbook above) on how to [set up your environment to take TensorFlow Developer Certification Exam](https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf).

Again, I'm not going to repeat what's mentioned in the document too much because it's another well-written guide (plus, if things change over time, new versions etc, best to adhere to the guide).

Reading through this as well as following each of the tests it suggests will ensure your computer is ready to go.

🛠 **Exercises**

1. Go through the [PyCharm quick start](https://www.jetbrains.com/pycharm/learning-center/) tutorials to make sure you're familiar with PyCharm (the exam uses PyCharm, you can download the free version).
2. Read through and follow the suggested steps in the [setting up for the TensorFlow Developer Certificate Exam guide](https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf).
3. After going through (2), go into PyCharm and make sure you can train a model in TensorFlow. The model and dataset in the example `image_classification_test.py` [script on GitHub](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_classification_test.py) should be enough. If you can train and save the model in under 5-10 minutes, your computer will be powerful enough to train the models in the exam.
    - Make sure you've got experience running models locally in PyCharm before taking the exam. Google Colab (what we used through the course) is a little different to PyCharm.

![before taking the TensorFlow Developer certification exam, make sure you can run TensorFlow code in PyCharm on your local machine](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/11-getting-example-script-to-run-in-pycharm.png)
*Before taking the exam make sure you can run TensorFlow code on your local machine in PyCharm. If the [example `image_class_test.py` script](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_classification_test.py) can run completely in under 5-10 minutes on your local machine, your local machine can handle the exam (if not, you can use Google Colab to train, save and download models to submit for the exam).*

## Troubleshooting tidbits

If you've been through the Zero to Mastery TensorFlow for Deep Learning course, you've had plenty of experience troubleshooting different models. 

But for reference, here are some of the main issues you might run into (inside and outside of the exam):

- **Input and output shapes** — print these out if you're stuck.
- **Input and output datatypes** — TensorFlow usually prefers float32.
- **Output activation functions** — for classification: `sigmoid` vs `softmax`, which one should you use?
- **Loss functions** — for classification `sparse_categorical_crossentropy` vs `categorical_crossentropy`, which one should you use?
- **Ways to improve a model** — if your model isn't performing as well as it should, what can you do?

## Questions

**Can I use external resources (Google, Stack Overflow, TensorFlow documentation, previous code) during the exam?**

Yes. The exam is open book. You will have access to all of the resources you would usually have access to whilst writing TensorFlow code outside of the exam.

**Do I need a GPU?**

No. The models built in the exam aren't extremely large. So you can train them on CPU. I'd advise trying out to see if you can run one of the example models and if you can train in under 5-10 minutes (without GPU), you can proceed with the exam. If you need access to a GPU, you can always train a model on Google Colab and download it in `.h5` and then submit it during the exam.

**Can I just do the courses, read the book(s) and practice myself, do I really need the certificate?**

Of course. At the end of the day, skills are what you should be after, not certificates. Certificates are *nice* to haves not *need* to haves.

**If you say certificates aren’t needed, why’d you get it?**

I like having a challenge to work towards. Setting a date for myself, as in, “I’m taking the exam on June 3rd”, gave me no choice but to study.

## Extra-curriculum

If you'd like some extra materials to go through to further your skills with TensorFlow and deep learning in general or to prepare more for the exam, I'd highly recommend the following:

- [TensorFlow in Practice Specialization on Coursera](https://dbourke.link/tfinpractice)
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow 2nd Edition](https://amzn.to/3aYexF2)
- [MIT Introduction to Deep Learning](http://introtodeeplearning.com/)
