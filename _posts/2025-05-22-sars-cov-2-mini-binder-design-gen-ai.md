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

1. **Folding prediction with Boltz2** to ensure structural viability and stability of the designed proteins.

1. **Docking** to the SARS-CoV-2 spike RBD using LightDock, to assess binding potential at the ACE2 interface.

The final result is a collection of entirely novel mini-proteins, *i.e.*, short sequences that are predicted to fold into compact, stable structures and potentially bind to the viral RBD. This project demonstrates not only the power of modern AI tools in synthetic biology, but also how individuals outside of traditional wet labs can now explore therapeutic design using open data and open models.

In the rest of this post, I’ll walk you through each step in the pipeline, share design choices and results, and reflect on the challenges and opportunities in this emerging space.

## 1. Backbone generation: RFDiffusion with ACE2 Hotspot constraints

The first step in the pipeline was to generate candidate binder backbones that are geometrically compatible with the SARS-CoV-2 spike RBD, and in particular with the ACE2 binding site. For this, I used RFdiffusion, a diffusion-based generative model for protein structures that can build new backbones in the context of a fixed target protein and optional interface constraints (“hotspots”).

Defining the design problem

I started from a high-resolution structure of the SARS-CoV-2 RBD bound to human ACE2, focusing specifically on the region of ACE2 that contacts the viral RBD. To bias the designs toward biologically meaningful interactions, I relied on the hotspot analysis from Veeramachaneni et al. (“Structural and simulation analysis of hotspot residues interactions of SARS-CoV-2 with human ACE2 receptor”).

From this study, I selected six ACE2 hotspot residues that contribute strongly to RBD binding:

1. THR500

1. GLY502

1. TYR505

1. GLN498

1. GLN493

1. ASN487

These residues sit at the heart of the ACE2–RBD interface, forming a dense network of hydrogen bonds and van der Waals contacts with the spike protein. In RFdiffusion, I treated them as target-side hotspots: positions on the RBD surface where the model is encouraged to place complementary binder residues.

![ACE2 Spike protein RBD interface](https://beautifuljekyll.com/assets/img/Figure1.jpg)
*Figure 1: ACE2 receptor - Spike Protein RBD interaction interface with highlighted hotspots selected for RFDiffusion design.*

### RFdiffusion setup

Conceptually, the RFdiffusion setup looked like this:

- The RBD structure is held fixed as the “target”.

- A new binder chain is initialized near the ACE2 interface and allowed to diffuse (i.e., evolve) into a structured backbone.

- The hotspot residues on the RBD are passed to the model so that it preferentially forms contacts in their vicinity.

Rather than copying the ACE2 sequence or structure directly, the idea was to let RFdiffusion invent entirely new scaffolds that nonetheless “hug” the same region of the RBD that ACE2 uses. In practice, this means we’re asking the model:

“Generate a small, stable backbone that packs against this specific patch of the RBD and makes good contacts with these hotspot residues.”

I ran 10 independent RFdiffusion trajectories, each allowed to generate a candidate mini-protein backbone in the presence of the RBD and hotspot constraints.

### What RFdiffusion actually produced

The 10 trajectories naturally fell into two qualitative categories:

**Single-chain designs with a dangling tail**

  In most runs, RFdiffusion produced a single, relatively long chain that contained a compact interface region plus an extended segment sticking out into solvent. That extra segment was clearly non-interacting: no meaningful contacts with the RBD, just a flexible-looking tail. While these are technically valid structures, they are not ideal as mini-protein binders: they’re larger than necessary and potentially less stable and harder to express.

**Compact two-helix designs at the interface**

In three runs (design 1, 4, and 7), RFdiffusion generated short, compact backbones made of two α-helices arranged against the RBD surface. These helices qualitatively resemble the ACE2 helical segment that engages the spike protein, but are not copies—they’re de novo scaffolds that happen to occupy a similar part of structural space. Importantly, in these designs the helices sit directly over the ACE2–RBD interface, making close approach to the hotspot residues defined above.

![Single chain design](https://beautifuljekyll.com/assets/img/Figure2.jpg)
*Figure 2: Several designs were a single relatively long alpha helix. Most of the protein sticking out into solvent, not interacting with the spike protein*


### Selecting the top backbones

For a first pass, I used simple, qualitative criteria to select promising candidates:

- **Compactness**: no long unstructured tails or obvious “dead” regions far from the RBD.

- **Interface placement**: the backbone should sit snugly over the ACE2 binding patch on the RBD.

- **Hotspot engagement (visual)**: the generated helices should appear to make contacts across the hotspot region, rather than drifting to some unrelated patch of the surface.

Based on these criteria, I discarded the “single long chain with a dangling tail” designs and kept designs 1, 4, and 7 as the top three backbones for further work. These three candidates provided the right balance of: small size (mini-protein-like), clear interface geometry, and ACE2-like orientation at the RBD surface.

In the next step, I took these three backbones into ProteinMPNN to design amino acid sequences that could plausibly fold into these shapes while maintaining their interface contacts to the spike RBD.

## 2. Inverse folding with ProteinMPNN: from structure to sequence

With the three most promising RFdiffusion backbones in hand (designs 1, 4, and 7), the next step was inverse folding: assigning amino acid sequences that are compatible with each backbone’s 3D geometry. For this, I used ProteinMPNN (pMPNN), a deep learning model trained to predict realistic protein sequences given fixed backbone coordinates.

Conceptually, this step answers a simple but crucial question:

_If this backbone were a real protein, what sequences could stably fold into it?_

### ProteinMPNN setup

I ran ProteinMPNN using its default parameters, without adding any residue-level constraints or biases. The goal here was not to over-engineer the design, but to let the model freely explore sequence space given the structural information encoded in the backbone.

A deliberate simplification in this stage was that I did not force the first residue to be methionine (M). From a biological and experimental perspective, this would of course be required for expression in most systems.

However, this project is entirely in silico, and at this stage, I was primarily interested in validating the pipeline logic, rather than producing expression-ready constructs.

In a follow-up iteration aimed at wet-lab validation, this constraint could easily be added either directly in pMPNN or later during construct design.

### Generating sequence diversity

For each of the three selected backbones, I sampled 4 independent sequences, resulting in a total of 12 candidate mini-protein binders.

Sampling multiple sequences per backbone is important for two reasons:

- **Robustness**, if a backbone is viable, there should be many sequences that can realize it, not just one.

- **Interface exploration**, different side-chain patterns on the same scaffold can lead to different interaction modes with the RBD, potentially affecting affinity and specificity.

Even without explicit interface constraints, ProteinMPNN tends to produce sequences with sensible biophysical properties: compact hydrophobic cores within the helices, and more polar or charged residues exposed to solvent and the binding interface.

### Scope and limitations

At this stage, the output sequences should be viewed as candidate realizations of the designed backbones, not optimized binders. I did not:

- Enforce specific residue identities at the RBD interface

- Bias toward known ACE2-like motifs

- Optimize for expression, solubility, or aggregation resistance

All of these would be natural next steps in a more mature design cycle. Here, the objective was simpler: verify that clean, compact two-helix backbones produced by RFdiffusion can be populated by realistic sequences using an off-the-shelf inverse folding model.

In the next section, I move from sequence design to structural validation, using Boltz2 (as an AlphaFold alternative) to check whether these sequences actually fold back into their intended geometries and remain compatible with the original binder design.

## 3. Folding validation with Boltz2: good vs bad alignments

With sequences in hand, the next step was to ask a crucial question: do these designs actually fold the way they’re supposed to? To answer this, I used Boltz2 for structure prediction.

For each of the 12 designed sequences (4 per backbone × 3 backbones), Boltz2 produced four structural models, saved as model_0 through model_3. I won’t speculate here on the exact meaning of these model indices, but in practice they represent alternative folding hypotheses for the same sequence.

What matters is that not all models are equally good.

Comparing predicted folds to the designed backbone

To evaluate the results, I aligned each Boltz2-predicted structure to its original RFdiffusion backbone and visually inspected the overlap. This allowed me to distinguish between two very different outcomes:

Good folding predictions, where the mini-protein adopts a structure that closely matches the designed backbone and sits in the expected position relative to the RBD.

Bad folding predictions, where the protein is still folded but is dramatically mispositioned, often rotated or shifted away from the intended binding interface.

Rather than exhaustively showing all 48 Boltz2 outputs (12 sequences × 4 models), I selected two representative examples to illustrate the contrast.

A “good” alignment

In the first example, the predicted structure overlays cleanly with the original backbone:

The two helices are preserved.

The overall geometry matches the design.

The RMSD is below 0.9 Å, with only minor local displacements.

These small deviations are entirely expected and even reassuring: they reflect the natural flexibility of helices and side-chain packing differences, rather than a failure of the design. Qualitatively, this is exactly what you hope to see when validating a de novo mini-protein: the sequence folds back into the intended scaffold.

A “bad” alignment

In the second example, the contrast is stark. Although the protein still forms secondary structure, the binder is dramatically rotated relative to the expected position. When aligned to the same reference:

The backbone no longer overlaps with the designed scaffold.

The helices are displaced away from the ACE2-like interface.

The binder is clearly incompatible with productive binding to the RBD.

This kind of failure mode is important to highlight. It shows that a predicted fold can look “reasonable” in isolation, yet still be completely wrong in the context of a binding task.

Why this comparison matters

These two figures are not meant to be statistically representative or exhaustive. Instead, they serve a simpler purpose:

To visually define what success and failure look like in this pipeline.

A low RMSD, clean overlap, and correct orientation relative to the target are strong indicators that the sequence–structure pair is viable. Large rotations or displacements, even if the protein appears folded, are a red flag for downstream binding and docking steps.

This step turned out to be a critical filter in the pipeline: before worrying about docking scores or interface energetics, it’s essential to first confirm that the designed sequences can reliably realize the intended backbone geometry.

In the next section, I’ll build on this validation step and show how these filtered designs were taken forward for docking and interface assessment against the SARS-CoV-2 spike RBD.






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
