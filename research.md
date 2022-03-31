# Papers

# Self driving
## 2022
- [End-to-end deep learning-based autonomous driving control for high-speed environment](https://link.springer.com/article/10.1007/s11227-021-03929-8)
- [Design of robust deep learning-based object detection and classification model for autonomous driving applications](https://link.springer.com/article/10.1007/s00500-021-06706-0)
- [Review the state-of-the-art technologies of semantic segmentation based on deep learning](https://www.sciencedirect.com/science/article/pii/S0925231222000054?casa_token=d0LOQT-xlBcAAAAA:0KkE_saxoMvDfSuf3-CpHxCHYtn8SGNyHy_17k-mcp0y-SXmD2IS75q3XRsgaqMPdXPl-S9Y)
  - On semantic segmentation. References self-driving applications
- [Deep Learning-Based Modeling of Pedestrian Perception and Decision-Making in Refuge Island for Autonomous Driving](https://link.springer.com/chapter/10.1007/978-3-030-77185-0_9)
- [Pyramid Bayesian Method for Model Uncertainty Evaluation of Semantic Segmentation in Autonomous Driving](https://link.springer.com/article/10.1007/s42154-021-00165-x)
- [Efficient Deep Reinforcement Learning With Imitative Expert Priors for Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/9694460?casa_token=PlOaQ1K5ZL0AAAAA:5VKp4RkPXnAc6Vi1tlmaBVK8QyoKEm18mNHzVggzB6Y2HJPnsCkKVFAH2KdjDnF1c2YIIc0)
- [Combining YOLO and Deep Reinforcement Learning for Autonomous Driving in Public Roadworks Scenarios](https://www.scitepress.org/Papers/2022/109136/109136.pdf)
## 2021
- [Deep Learning for Safe Autonomous Driving: Current Challenges and Future Directions](https://ieeexplore.ieee.org/abstract/document/9284628)
## 2020
- [A survey of deep learning techniques for autonomous driving](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21918?casa_token=HQXYv2N9VWsAAAAA:EDhhQnfs2misG54n-sOS3JsT_Qpidp_8GPjtAWcM8O-Tp64rbscgNXpdGyLuo9A6bxTSUWV7dQuk)
## 2019
- [Toward a Brain-Inspired System: Deep Recurrent Reinforcement Learning for a Simulated Self-Driving Agent
](https://www.frontiersin.org/articles/10.3389/fnbot.2019.00040/full)
- [Deep learning-based image recognition for autonomous driving](https://www.sciencedirect.com/science/article/pii/S0386111219301566)
## 2017
- [Deep Reinforcement Learning framework for Autonomous Driving](https://www.ingentaconnect.com/content/ist/ei/2017/00002017/00000019/art00012)





# GAN 
- [Inverse Graphics GAN: Learning to Generate 3D Shapes from Unstructured 2D Data](https://arxiv.org/pdf/2002.12674.pdf)



# Image Classification
## OpenAI CLIP
https://blog.roboflow.com/openai-clip/
https://blog.roboflow.com/how-to-use-openai-clip/

## Use-Case:
- Feed in original image => get text representation of image.
  - For example, if the input is an image of dog on a lawn, the output is "image of a dog on a lawn".

# Image Generation
## OpenAI DALL-E
https://openai.com/blog/dall-e/
https://github.com/openai/dall-e

## Use-Case:
- The user edits the original text representation => this model generates a new image.
  - For example, "image of a dog on a street" => model generates corresponding image

## Performance and Limitations

The heavy compression from the encoding process results in a noticeable loss of detail in the reconstructed images. This
renders it inappropriate for applications that require fine-grained details of the image to be preserved.


# Libraries To Use

- [Rendering 3D Images with Pytorch3d](https://towardsdatascience.com/how-to-render-3d-files-using-pytorch3d-ef9de72483f8)

# Conference Papers:
- [CVF Sponsored Conferences](https://openaccess.thecvf.com/menu)
- [WACV 2022](https://openaccess.thecvf.com/WACV2022)
  - [UNETR: Transformers for 3D Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf)
  - [GraN-GAN: Piecewise Gradient Normalization for Generative Adversarial
Networks](https://openaccess.thecvf.com/content/WACV2022/papers/Bhaskara_GraN-GAN_Piecewise_Gradient_Normalization_for_Generative_Adversarial_Networks_WACV_2022_paper.pdf)
  - [Pixel-Level Bijective Matching for Video Object Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf)
  - [Multimodal Learning using Optimal Transport
for Sarcasm and Humor Detection](https://openaccess.thecvf.com/content/WACV2022/papers/Pramanick_Multimodal_Learning_Using_Optimal_Transport_for_Sarcasm_and_Humor_Detection_WACV_2022_paper.pdf)
  - [Semi-Supervised Semantic Segmentation of Vessel Images Using Leaking Perturbations](https://openaccess.thecvf.com/content/WACV2022/papers/Hou_Semi-Supervised_Semantic_Segmentation_of_Vessel_Images_Using_Leaking_Perturbations_WACV_2022_paper.pdf)
- [CVPR 2021](https://openaccess.thecvf.com/CVPR2021?day=all)
  - [Zero-Shot Instance Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Zero-Shot_Instance_Segmentation_CVPR_2021_paper.pdf)
  - [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual
  Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Propagate_Yourself_Exploring_Pixel-Level_Consistency_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf)
    - Microsoft Research Asia
  - [Exploring Simple Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
    - Facebook AI Research 
  - [Pre-Trained Image Processing Transformer](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)
- 
## [NVIDIA's New AI: Enhance!](https://www.youtube.com/watch?v=e0yEOw6Zews)

- 📝 The paper "EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks" is available here:
  https://matthew-a-chan.github.io/EG3D/
- 📝 The latent space material synthesis paper "Gaussian Material Synthesis" is available here:
  https://users.cg.tuwien.ac.at/zsolnai...

## [NVIDIA’s New AI: Wow, Instant Neural Graphics! 🤖](https://www.youtube.com/watch?v=j8tMk-GE8hY)

- 📝 #NVIDIA's paper "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" is available here:
  https://nvlabs.github.io/instant-ngp/

## [NVIDIA’s New AI: Superb Details, Super Fast! 🤖](https://www.youtube.com/watch?v=eaSTGOgO-ss&t=1s)

- 📝 The paper "Multimodal Conditional Image Synthesis with Product-of-Experts GANs" (#PoEGAN) is available here:
  https://deepimagination.cc/PoE-GAN/

## [Next Level Paint Simulations Are Coming! 🎨🖌️](https://www.youtube.com/watch?v=b2D_5G_npVI)

- 📝 The paper "Practical Pigment Mixing for Digital Painting" is available here:
  https://scrtwpns.com/mixbox/

## [NVIDIA’s New AI Draws Images With The Speed of Thought! ⚡](https://www.youtube.com/watch?v=Wbid5rvCGos&t=271s)

- 📝 The previous paper "Semantic Image Synthesis with Spatially-Adaptive Normalization" is available here:  
  https://nvlabs.github.io/SPADE/

## [New AI: Next Level Video Editing! 🤯](https://www.youtube.com/watch?v=MCq0x01Jmi0&t=192s)

- 📝 The paper "Layered Neural Atlases for Consistent Video Editing" is available here:        
  https://layered-neural-atlases.github.io/

## [Photos Go In, Reality Comes Out…And Fast! 🌁](https://www.youtube.com/watch?v=yptwRRpPEBM)

- 📝 The paper "Plenoxels: Radiance Fields without Neural Networks" is available here:         
  https://alexyu.net/plenoxels/
  [Google’s New AI: This is Where Selfies Go Hyper! 🤳](https://www.youtube.com/watch?v=B-zxoJ9o7s0&t=88s)
- 📝 The paper "A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields" is available
  here:      
  https://hypernerf.github.io/

## [Enhance! Super Resolution Is Here! 🔍](https://www.youtube.com/watch?v=WCAF3PNEc_c)

- 📝 The paper "Image Super-Resolution via Iterative Refinement " is available here:
  https://iterative-refinement.github.io/
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement

## [This Image Is Fine. Completely Fine. 🤖](https://www.youtube.com/watch?v=BS2la3C-TYc&t=354s)

- 📝 The paper "The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning"
  is available here:      
  https://attentionneuron.github.io/

## [NVIDIA’s New Technique: Beautiful Models For Less! 🌲](https://www.youtube.com/watch?v=ogL-2IClOug&t=151s)

- 📝 The paper "Appearance-Driven Automatic 3D Model Simplification" is available here:          
  https://research.nvidia.com/publication/2021-04_Appearance-Driven-Automatic-3D
- 📝 The differentiable material synthesis paper is available here:          
  https://users.cg.tuwien.ac.at/zsolnai/gfx/photorealistic-material-learning-and-synthesis/

## [Finally, Video Stabilization That Works! 🤳](https://www.youtube.com/watch?v=v5pOsQEOsyA&t=155s)

- 📝 The paper "FuSta - Hybrid Neural Fusion for Full-frame Video Stabilization" is available here:
	- Paper https://alex04072000.github.io/FuSta/
	- Code: https://github.com/alex04072000/FuSta
	- Colab: https://colab.research.google.com/drive/1l-fUzyM38KJMZyKMBWw_vu7ZUyDwgdYH?usp=sharing

## [This AI Learned To Stop Time! ⏱](https://www.youtube.com/watch?v=4CYI6dt1ZNY)

- 📝 The paper "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes" is available
  here: http://www.cs.cornell.edu/~zl548/NSFF/

## [This AI Makes Beautiful Videos From Your Images! 🌊](https://www.youtube.com/watch?v=t7nO7MPcOGo&t=139s)

- 📝 The paper "Animating Pictures with Eulerian Motion Fields" is available here:
  https://eulerian.cs.washington.edu/

## [3 New Things An AI Can Do With Your Photos!](https://www.youtube.com/watch?v=B8RMUSmIGCI&t=324s)

- 📝 The paper "GANSpace: Discovering Interpretable GAN Controls" is available here:
  https://github.com/harskish/ganspace

- 📝 Our material synthesis paper is available
  here: https://users.cg.tuwien.ac.at/zsolnai/gfx/gaussian-material-synthesis/

- 📝 The font manifold paper is available here: http://vecg.cs.ucl.ac.uk/Projects/projects_fonts/projects_fonts.html

## [What Is 3D Photography? 🎑](https://www.youtube.com/watch?v=BjkgyKEQbSM)

- 📝 The paper "One Shot 3D Photography" is available here: https://facebookresearch.github.io/one_shot_3d_photography/

## [Enhance! Neural Supersampling is Here! 🔎](https://www.youtube.com/watch?v=OzHenjHBBds)

- 📝 The paper "Neural Supersampling for Real-time Rendering" is available here:
  https://research.facebook.com/blog/2020/07/introducing-neural-supersampling-for-real-time-rendering/
  https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/

## [AI-Based Style Transfer For Video…Now in Real Time!](https://www.youtube.com/watch?v=UiEaWkf3r9A)

- 📝 The paper "Interactive Video Stylization Using Few-Shot Patch-Based Training" is available
  here: https://ondrejtexler.github.io/patch-based_training/

## [This AI Creates Real Scenes From Your Photos! 📷](https://www.youtube.com/watch?v=T29O-MhYALw)

- 📝 The paper "NeRF in the Wild - Neural Radiance Fields for Unconstrained Photo Collections" is available here:
  https://nerf-w.github.io/

## [OpenAI’s Image GPT Completes Your Images With Style!](https://www.youtube.com/watch?v=-6Xn4nKm-Qw)

- 📝 The paper "Generative Pretraining from Pixels (Image GPT)" is available here:
  https://openai.com/blog/image-gpt/

## [https://www.youtube.com/watch?v=qeZMKgKJLX4](https://www.youtube.com/watch?v=qeZMKgKJLX4)

- 📝 The paper "Portrait Shadow Manipulation" is available here:
  https://ceciliavision.github.io/project-pages/portrait

- 📝 Our paper with Activision Blizzard on subsurface scattering is available here:
  https://users.cg.tuwien.ac.at/zsolnai/gfx/separable-subsurface-scattering-with-activision-blizzard/

## [Can an AI Learn Lip Reading?](https://www.youtube.com/watch?v=wg3upHE8qJw)

- 📝 The paper "Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis" is available here:
  https://cvit.iiit.ac.in/research/projects/cvit-projects/speaking-by-observing-lip-movements

- Our earlier video on the "bag of chips" sound reconstruction is available here:
  https://www.youtube.com/watch?v=2i1hrywDwPo

## [This AI Creates Beautiful Time Lapse Videos ☀️](https://www.youtube.com/watch?v=EWKAgwgqXB4)

- 📝 The paper "High-Resolution Daytime Translation Without Domain Labels" is available here:

- https://saic-mdal.github.io/HiDT/
- https://github.com/saic-mdal/HiDT

## [Neural Network Dreams About Beautiful Natural Scenes](https://www.youtube.com/watch?v=MPdj8KGZHa0)

- 📝 The paper "Manipulating Attributes of Natural Scenes via Hallucination" is available here:
  https://hucvl.github.io/attribute_hallucination/

## [This AI Learned to Summarize Videos 🎥](https://www.youtube.com/watch?v=bVXPnP8k6yo)

- 📝 The paper "CLEVRER: CoLlision Events for Video REpresentation and Reasoning" is available here:
  http://clevrer.csail.mit.edu/
