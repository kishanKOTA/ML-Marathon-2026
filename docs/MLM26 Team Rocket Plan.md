### **Snapshot WI** \- Oh Deer 🦌

ML Marathon 2026 \
September 9 \- December 9 

---

Index

* [Quick Links](#quick-links-🔗)  
* [Logistics](#logistics-🗺️)  
  * [Communication](#communication)  
  * [Team Meetings](#team-meetings)  
* [Challenge](#challenge⚡)  
  * [Overview](#overview)  
  * [Evaluation](#evaluation)  
  * [Data](#data)  
  * [Observations](#observations)  
  * [Pre-modeling ideas](#ideas)  
* [AI Use Agreement](#ai-use-agreement)  
* [Misc](#heading=h.s5nbvxqqmbw1)

### ---

### **Quick Links 🔗** {#quick-links-🔗}

[Marathon Schedule](https://ml-marathon.wisc.edu/schedule/)  
[Kaggle Challenge Description](https://www.kaggle.com/competitions/snapshot-wi-oh-deer)  
[GitHub Repo](https://github.com/kishanKOTA/ML-Marathon-2026)  

---

### **Challenge⚡** {#challenge⚡}

#### Overview {#overview}

Use Snapshot Wisconsin trail camera images of deer to create a model that detects **age** and **antlers**.

Age

* **Young deer** have bright spots in their fur  
* **Adult deers** have no spots, unless caused by discoloration which is visually distinct

Antlers

* **Antlered deer** when larger than or equal to 3 inches (longer than ear)  
* **Antlerless deer** when smaller than 3 inches

Additionally, we can try to extract additional information such as time of year/day, the animal’s state (alert or relaxed),  land cover characteristics, etc. These additional predictions won’t have ground truth.

#### Evaluation {#evaluation}

We will be evaluated on:

* Accuracy measured by F1 Score \- \[100 points\]  
* Additional predictions beyond age and antlers \- \[15 points\]  
* Inference speed \- \[15 points\]  
* Efficiency based on compute  \- \[10 points\]  
* Ease of use \- \[10 points\]

We submit one Writeup as a team. We can edit and resubmit it as many times as we want before the deadline. The submission **deadline is December 2**

#### Data {#data}

4880 images of deer, a ground truth data file containing bounding boxes (Snapshot\_WI-Oh\_Deer\_data-v2.csv), and locations at the county level (Snapshot\_WI-Oh\_Deer\_locs-v2.csv).

#### Observations {#observations}

* Images during the day are colored, at night black and white  
* Motion blur seems to be common  
* Varying levels of lighting in day images


#### Pre-modeling ideas {#ideas}

1. Read Challenge Description - Understand the requirements and goals on the challenge on the Kaggle page. Summarize objectives and priorities in a shared team doc
2. Research prior work - Look at papers, articles, tutorials to identify common methods/models applied to this task 
3. Exploratory Data Analysis (EDA) - Look at the challenge’s images and get a sense of image and label format/dimensions
4. Set up repository and contribution guidelines - Including branch naming, 	PR size, commit messages, etc
5. Choose performance metrics and baselines - Determine what metrics to evaluate model performance and look at other camera trap benchmarks for comparison
6. Choose train-test-validate split - And organize images in the repo accordingly (maybe set up an automated way to fetch from Kaggle directly)

---

### **AI Use Agreement** {#ai-use-agreement}

We talked about how AI could generate code that would write the entire project for us, and how we want to use this marathon as an opportunity to learn. We also know that AI itself is a great tool, and there is value in using it to help us with the challenge. So these are some guidelines we will try to follow and adjust as we go:

Fine to use for ✅

* Planning, learning, and explanations of code or syntax  
* Troubleshooting code

Not fine for ❌

* Generating our entire training pipeline  
* Commit and merge PRs on our behalf