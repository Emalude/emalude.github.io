---
layout: post
title: Designing a Mini-Protein Binder Against SARS-CoV-2 RBD with Generative AI
subtitle: Using RFDiffusion and proteinMPNN
gh-repo: emalude/sars-cov2-binder-design
gh-badge: [star, fork, follow]
tags: [Project, GenAI]
comments: true
mathjax: true
author: Emanuele Sicurella
---
The rapid rise of protein language models and generative AI has opened the door to a bold new frontier in synthetic biology: **designing novel proteins from scratch using only computational tools**. Among the most compelling applications is the creation of **mini-protein binders**: small, stable proteins that can latch onto a biological target with high specificity, offering potential as therapeutics, diagnostics, or research tools.

In this project, I set out to design a **de novo mini-protein binder for the SARS-CoV-2 spike receptor-binding domain (RBD)**, the same region that allows the virus to enter human cells by interacting with the ACE2 receptor. This interface has been extensively studied and serves as a well-defined benchmark for computational protein design.

But rather than relying on known binders or sequence motifs, my goal was more ambitious: to build a binder from scratch, using an end-to-end generative pipeline made entirely of open-source tools. The project draws inspiration from cutting-edge techniques like **diffusion-based structure generation, sequence design with deep learning, and structure prediction via AlphaFold2**, all culminating in in silico docking and structural validation.

Here’s the core pipeline I implemented:

1. **Backbone generation** using RFdiffusion, a generative model for protein structures.

1. **Sequence design** for the generated backbones using ProteinMPNN, a deep learning model that maps structures to likely sequences.

1. **Folding prediction with AlphaFold2 via ColabFold** to ensure structural viability and stability of the designed proteins.

1. **Docking** to the SARS-CoV-2 spike RBD using tools like HADDOCK or ZDOCK, to assess binding potential at the ACE2 interface.

The final result is a collection of entirely novel mini-proteins, *i.e.*, short sequences that are predicted to fold into compact, stable structures and potentially bind to the viral RBD. This project demonstrates not only the power of modern AI tools in synthetic biology, but also how individuals outside of traditional wet labs can now explore therapeutic design using open data and open models.

In the rest of this post, I’ll walk you through each step in the pipeline, share design choices and results, and reflect on the challenges and opportunities in this emerging space.



{: .box-success}
This is a demo post to show you how to write blog posts with markdown.  I strongly encourage you to [take 5 minutes to learn how to write in markdown](https://markdowntutorial.com/) - it'll teach you how to transform regular text into bold/italics/tables/etc.<br/>I also encourage you to look at the [code that created this post](https://raw.githubusercontent.com/daattali/beautiful-jekyll/master/_posts/2020-02-28-sample-markdown.md) to learn some more advanced tips about using markdown in Beautiful Jekyll.

**Here is some bold text**

## Here is a secondary heading

[This is a link to a different site](https://deanattali.com/) and [this is a link to a section inside this page](#local-urls).

Here's a table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |

You can use [MathJax](https://www.mathjax.org/) to write LaTeX expressions. For example:
When \\(a \ne 0\\), there are two solutions to \\(ax^2 + bx + c = 0\\) and they are $$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

How about a yummy crepe?

![Crepe](https://beautifuljekyll.com/assets/img/crepe.jpg)

It can also be centered!

![Crepe](https://beautifuljekyll.com/assets/img/crepe.jpg){: .mx-auto.d-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.

## Local URLs in project sites {#local-urls}

When hosting a *project site* on GitHub Pages (for example, `https://USERNAME.github.io/MyProject`), URLs that begin with `/` and refer to local files may not work correctly due to how the root URL (`/`) is interpreted by GitHub Pages. You can read more about it [in the FAQ](https://beautifuljekyll.com/faq/#links-in-project-page). To demonstrate the issue, the following local image will be broken **if your site is a project site:**

![Crepe](/assets/img/crepe.jpg)

If the above image is broken, then you'll need to follow the instructions [in the FAQ](https://beautifuljekyll.com/faq/#links-in-project-page). Here is proof that it can be fixed:

![Crepe]({{ '/assets/img/crepe.jpg' | relative_url }})

<details markdown="1">
<summary>Click here!</summary>
Here you can see an **expandable** section
</details>
