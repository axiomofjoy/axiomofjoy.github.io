---
layout: post
title: "1000 Genomes: Part I"
subtitle: "Unsupervised learning, dimensionality reduction, and visualization of genomic data."
author: "Alexander Song"
background: '/media/posts/genome_part_i/dna-3539309.jpg'
extra_css: '/assets/post_css/onekgenome1.css'
---


# 1000 Genomes: Part I

This post introduces the linear and non-linear dimensionality reduction techniques of dual and kernel PCA in the context of genomic data. I will describe the mathematics behind these techniques and will then apply them to visualize high-dimensional genomic data in two and three dimensions. The resulting low dimensional representations provide insight into original data; in particular, distinct gender, ethnic, and racial groups appear as discernible clusters, while groups of mixed heritage are situated between . In the next post, I'll use SVMs and neural networks to predict an individual's ethnic origin based on his or her genome.

## Previous Work

This post was inspired by the most fascinating application of PCA I have yet to encounter, a 2009 study which studied the genomes of 3,192 European individuals at 500,568 loci [FIXME]. Remarkably, a PCA of the genomic data using the top two principal component resulted in a high-resolution map of Europe, reproducing geographic features such as the Iberian and Italian peninsulas and the British Isles and even resolving sub-populations within individual countries (see Figure 1 below).

<figure>
  <img src="/media/posts/genome_part_i/europe_map.png" alt="PCA of genomic data reproduces map of Europe" style="width:100%">
  <div class="caption">Figure 1: A PCA of European genomic data using the top two principal components results in a map of the European subcontinent.</div>
</figure>

The data set used in the study is unfortunately not publicly available. However, Amazon Web Services hosts the 1000 Genomes Project, containing genomes sequenced at roughly 81 million loci from 2504 individuals belonging to 26 different population groups around the world, as a publicly available S3 Bucket. The high dimension of the feature space (in this case, the number of sequenced genes) precludes the possibility of applying standard techniques to compute the PCA of the data. Others have used more sophisticated techniques to compute PCA for the data set; for example, [this page](https://github.com/bwlewis/1000_genomes_examples) uses genomic data to demonstrate the merits of IRLBA, an efficient algorithm for computing truncated SVDs of large sparse and dense matrices. I will instead use the simpler (but less numerically stable) method of dual PCA in addition to the non-linear method of kernel PCA to reduce the dimension of the data and visualize it in two and three dimensions.


## Data Set

The 1000 Genomes Project was an international effort commencing in 2008 and completed in 2015 to catalogue human genetic variation. The project sequenced the genomes of 2504 individuals from 26 different population groups from around the world, and the resulting data set was made publicly available for the benefit of future research. Each individual's genome was sampled at roughly 81 million so-called "variant" sites, i.e., genes that frequently vary between different individuals, resulting in a high-dimensional feature space. Since different individuals share the vast majority of their genomes in common, genomic data is stored in so-called "variant call format" (VCF), which records the differences between each individual genome and a reference genome. To illustrate, here's a small excerpt consisting of two rows from a VCF file from the 1000 Genomes Project (for clarity, I have omitted most columns and have formatted the text so that columns are properly aligned).

```
#CHROM  POS     REF     ALT     HG00096 HG00097 HG00099 HG00100 HG00101
1       10177   A       AC      1|0     0|1     0|1     1|0     0|0
```

The second row above corresponds to a gene located in the first chromosome at position 10177. The reference genome has the nucleic acid adenine at this site, while certain individuals have the alternate adenine-cytosine gene. Codes uniquely identify each individual in the study (e.g., "HG00096"). Below these ID codes, the digits to the left and right of the pipe signify whether the reference or alternate gene are present on the left and right alleles of the corresponding sample, with zero corresponding to the reference gene and one corresponding to the alternate gene. For example, sample HG00096 has the variant gene AC on the left allele and the reference gene A on the right allele at position 10177 of the first chromosome. In certain cases (not shown above), multiple alternate genes may be given, in which case positive digits indicate which variant is present. This format applies for chromosomes 1 through 21 and the X chromosome. Since human beings possess at most one allele for the Y chromosome, the VCF entries for this chromosome consist of single digits rather than digits separated by a pipe. In rare cases, a period indicates that data is missing for a particular individual and site.

### Processing the Data Set


By recording the sites at which an individual genome differs from the reference genome, variant call format takes advantage of the highly redundant nature of the human genome and lends itself naturally to a sparse matrix representation of the data. For each chromosome, I downloaded the corresponding VCF file from the 1000 Genomes S3 Bucket to an EC2 instance. I then parsed the file with a simple C program that iterates over positions in the genome and individual samples. At each position and for each individual, I ignored which particular variant occurred, instead recording only whether a variant occurred at all. For example, the data for first chromosome is stored in FIXME sparse matrices. Each matrix has 2504 rows corresponding to the samples in the data set, and the combined number of columns is just the number of studied sites in the first chromosome. The _(i, j)<sup>th</sup>_ entry of a matrix is 0 if the _i<sup>th</sup>_ sample matches the reference genome (on both alleles) at the site corresponding to the _j<sup>th</sup>_ column, and 1 if a variant occurs. This format is relatively space efficient, since the sparse matrices have Boolean rather than integer entries. Even so, the entire genomes of all 2504 individuals required roughly eight gigabytes of compressed sparse matrices.

Each of the supervised and unsupervised learning techniques I apply in this post and the next may be implemented using a pairwise distance matrix, so I elected to compute this matrix first. Since the data is binary, Manhattan distance is the same as squared Euclidean distance, i.e., \\(\lVert \mathbf x - \mathbf y \rVert_1 = \lVert \mathbf x - \mathbf y \rVert_2^2\\) for all samples \\(\mathbf x\\) and \\(\mathbf y\\). The Manhattan distance between two genomes has an attractive interpretation in the current context; it simply counts the number of sites at which one genome has the reference gene and the other genome has a variant. Moreover, since Manhattan distance and squared Euclidean distance are equal, we can efficiently compute the Manhattan pairwise distance matrix for each of the FIXME sparse matrices containing a portion of the data set using a vectorized implementation like the one found [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html). Finally, the pairwise distance matrix for the entire data set can be easily computed by simply summing the pairwise distance matrices of each constituent sparse matrix.

## Dimensionality Reduction and Data Visualization

The explanation of dual and kernel PCA provided below assumes a familiarity with principal component analysis, singular value decomposition, and eigendecomposition.

### Dual PCA

Suppose a data set is represented by the matrix \\(X \in \mathbb R^{m \times n}\\), where \\(m\\) is the number of samples in the data set and \\(n\\) is the dimension of the feature space. To compute a standard PCA, we first center the data to obtain the matrix \\(X_c\\) with column means equal to zero. The principal components of the data are the right eigenvectors of the sample covariance matrix \\(\frac{1}{m-1}X_c^T X_c \in \mathbb R^{n \times n}\\), or equivalently, the right singular vectors of \\(X_c\\). The projection of the data onto the top \\(k\\) principal components is given by
\\begin{equation} \label{eq:pca_proj}
U_k \Sigma_k = X_c V_k,
\\end{equation}
where \\(\Sigma \in \mathbb R^{k \times k}\\) is the diagonal matrix containing the top \\(k\\) singular values of \\(X_c\\) in descending order, \\(U_k \in \mathbb R^{m \times k}\\) is the matrix of corresponding left singular vectors, and \\(V_k\\) is the matrix of the top \\(k\\) principal components. Thus, PCA can be computed using either the singular value decomposition of \\(X_c\\) or the eigendecomposition of \\(X_c^T X_c\\).


In the current context, the number of features \\(n \approx 81\\) million far exceeds the number of samples \\(m = 2504\\). Due to the high dimension of the feature space, the techniques described above fail since we cannot explicitly compute the sample covariance matrix (let alone its eigendecomposition), nor can we compute the SVD of \\(X_c\\) with a standard implementation. Since the number of samples \\(m = 2504\\) is relatively modest, we can instead use _dual PCA_, which essentially amounts to applying the so-called "kernel trick" with a linear kernel. Let's consider the relatively small _Gram matrix_ \\(X_c X_c^T \in \mathbb R^{m \times m}\\). If the SVD of \\(X_c\\) is given by \\(X_c = U \Sigma V^T\\) for unitary matrices \\(U \in \mathbb R^{m \times m}\\) and \\(V \in \mathbb R^{n \times n}\\) and diagonal matrix \\(\Sigma \in \mathbb R^{m \times n}\\), then the Gram matrix is
\\[
X_c X_c^T = U \hat{\Sigma}^2 U^T,
\\]
where \\(\hat{\Sigma}^2 \in \mathbb R^{m \times m}\\) consists of the first \\(m\\) columns of \\(\Sigma\\). In other words, we can recover the singular values and left singular vectors in \\eqref{eq:pca_proj} by computing the eigendecomposition of the Gram matrix.

There's one additional caveat. In the current case, we cannot form \\(X_c\\) since explicitly centering the data in \\(X\\) would destroy its sparse structure. Fortunately, I've already computed the pairwise squared Euclidean distance matrix \\(D_{\mathrm{sq}}\\) for \\(X\\). We can easily compute the Gram matrix with the formula
\\[
X_c X_c^T = -\left(I_m - \dfrac{\mathbf{1}\_m \mathbf{1}\_m^T}{m}\right) \frac{D_{\mathrm{sq}}}{2} \left(I_m - \dfrac{\mathbf{1}\_m \mathbf{1}\_m^T}{m}\right),
\\]
where \\(\mathbf{1}\_m \in \mathbb R^m\\) denotes the _one-vector_ whose entries are all one, and the outer product \\(\mathbf{1}\_m \mathbf{1}\_m^T\\) is the \\(m \times m\\) matrix whose entries are all one (see REF for a derivation).

While the explanation above is somewhat long-winded, the actual implementation is extremely simple (note that the Gram matrix is positive semi-definite, so the SVD is the same as the eigendecomposition). You can view the resulting PCA in two and three dimensions in the <a href="#visualization">visualization section</a>.

```python
import numpy as np
from numpy.linalg import svd

def dual_pca(D_sq, k):
    m = D_sq.shape[0]
    gram = -(np.eye(m) - 1 / m) @ D_sq @ (np.eye(m) - 1 / m)
    U, Sigma_sq, _ = svd(gram)
    Sigma_k = np.sqrt(Sigma_sq[:k])
    return U[:,:k] * Sigma_k
```

As explained in this [enlightening Stack Exchange post](https://stats.stackexchange.com/questions/14002/whats-the-difference-between-principal-component-analysis-and-multidimensional/132731#132731), this procedure to compute PCA is equivalent to classical multidimensional scaling using Euclidean distances.


<!-- Kernel PCA -->
<h3><a id="kernelPCA">Kernel PCA</a></h3>

Now that we've reduced the dimension of the data using the linear technique of dual PCA, I'll introduce a non-linear dimensionality reduction technique known as _kernel PCA_. A rigorous development of kernel methods is somewhat involved and is beyond the scope of this post. Instead, I'll provide an intuitive (and somewhat handwavy) explanation by way of comparison with the preceding section. As I previously hinted, dual PCA is simply kernel PCA with a so-called "linear kernel," and thus we have already seen kernel PCA in its simplest form.

Kernel methods are typically motivated as a means of reaping the benefits of high-dimensional feature spaces without incurring the associated computational costs. The following canonical example seeks to embed the XOR data set \\(\mathcal X\\) of points in \\(\mathbb R^2\\) into \\(\mathbb R\\).

<figure>
        <div class="caption">Figure 2: A toy example illustrating the merits of kernel PCA.</div>
    <div class="row">
        <div class="column">
            <img src="/media/posts/genome_part_i/data.png" alt="XOR data set" style="width:100%">
            <div class="caption">(a) The simple XOR data set is not linearly separable in \(\mathbb R^2\).</div>
        </div>
        <div class="column">
            <img src="/media/posts/genome_part_i/pca1.png" alt="A naive application of PCA." style="width:100%">
            <div class="caption">(b) A naive application of PCA "smashes" the original data.</div>
        </div>
    </div>
    <div class="row">
        <div class="column">
            <img src="/media/posts/genome_part_i/feature_map.png" alt="Mapping the data into higher dimensions." style="width:100%">
            <div class="caption">(c) The result of mapping the data set into \(\mathbb R^3\) using the non-linear map \((x, y) \mapsto (x^2, y^2, \sqrt 2 x y)\).</div>
        </div>
        <div class="column">
            <img src="/media/posts/genome_part_i/pca2.png" alt="A more reasonable application of PCA." style="width:100%">
            <div class="caption">(d) After mapping into higher dimensions, the data becomes linearly separable along the first principal component.</div>
        </div>
    </div>
</figure>


In the above example, we see that the original two-dimensional data is not linearly separable, and hence a naive application of PCA would effectively "smash" the data and destroy its structure. After applying a non-linear "feature map" \\(\phi\\) to map our data from the original space \\(\mathcal X = \mathbb R^2\\) into the higher dimensional space \\(\mathcal H = \mathbb R^3\\), the resulting higher dimensional representation of the data is linearly separable. We can then apply PCA to obtain an embedding into the real line that respects the non-linear structure of the original data. The moral of this example is that, in order to get a good lower-dimensional representation, we had to take the intermediate step of mapping the data into a higher-dimensional feature space.

The issue with the above technique is that the dimension of the feature space \\(\mathcal H\\) may be so high that it becomes impractical or even impossible to explicitly compute \\(\phi(\mathbf x)\\) for \\(\mathbf x \in \mathcal X\\). We encountered a similar issue in our discussion of dual PCA, when the high dimension of the original feature space precluded the possibility of applying standard PCA. In that situation, we were unable to explicitly calculate the principal components, which belonged to the original high-dimensional feature space. Instead, we projected the data onto the top \\(k\\) principal component and found the coordinates of this projection in the basis consisting of those top \\(k\\) principal components. To do so, all we needed was the Gram matrix (also known as the _linear kernel_), whose _(i,j)<sup>th</sup>_ entry is merely the inner product between the _i<sup>th</sup>_ and _j<sup>th</sup>_ data points.


The magic of kernels is that they allow us to compute inner products in high-dimensional feature spaces without the explicit computation of the feature map \\(\phi\\). The formal definition of a kernel invokes the notion of an _\\(\mathbb R\\)-Hilbert space_, a real inner product space \\(\mathcal H\\) such that \\(\mathcal H\\) is also a complete metric space with respect to the metric induced by the inner product. Loosely speaking, Hilbert spaces are spaces which have a well-defined notion of the angle between two elements in the space, and in which a few basic intuitions about distance hold true. A _kernel_ on the set \\(\mathcal X\\) is then a map \\(K: \mathcal X \times \mathcal X \rightarrow \mathbb R\\) such that there exists a _\\(\mathbb R\\)-Hilbert space_ \\(\mathcal H\\) and a _feature map_ \\(\phi: \mathcal X \rightarrow \mathcal H\\) satisfying
\\[
K(\mathbf x, \mathbf y) = \left\langle \phi(\mathbf x), \phi(\mathbf y) \right\rangle_{\mathcal H}
\\]
for all \\(\mathbf x, \mathbf y \in \mathcal H\\), where \\(\langle \cdot, \cdot \rangle_{\mathcal H}\\) denotes the inner product in \\(\mathcal H\\). Thus, by their very definition, kernels allow us to compute inner products in the (usually high-dimensional) feature space \\(\mathcal H\\) without incurring the cost of evaluating the feature map \\(\phi\\). As a further note, inner products in \\(\\mathcal H\\) are intimately related via the Law of Cosines to the notion of angles between elements in \\(H\\). Angles, in turn, can be thought of as a measure of "closeness" or "similarity." Thus, we can aptly describe kernels as computing the similarity between high-dimensional representations of elements in our data set.


Before introducing examples of specific kernels, I want to make a brief theoretical note. The above definition of kernels is motivated by a result from functional analysis known as _Mercer's Theorem_. While the details of the theorem are somewhat involved, the basic idea is quite simple; assuming that \\(\mathcal X\\) satisfies some technical conditions, then a symmetric continuous function \\(K: \mathcal X \times \mathcal X \rightarrow \mathbb R\\) is a kernel if and only if \\(K\\) is positive-definite.


In certain special cases, it's possible to explicitly describe the feature \\(\phi\\) corresponding to some kernel \\(K\\). More frequently, however, we simply verify that . We can then apply


Of course, the theory developed above is useful only insofar as there exist kernels corresponding to useful feature spaces and feature maps. We have already seen one such example in the familiar _linear kernel_. In the case of the linear kernel, the original feature space and _\\(\mathbb R\\)-Hilbert space_ are identical (i.e.,\\(\mathcal X = \mathcal H = \mathbb R^n\\)), the feature map \\(\phi\\) is the identity function on \\(\mathbb R^n\\), and the inner product \\(\rangle \cdot, \cdot \rangle_{\mathcal H}\\) is just the standard dot product on \\(\mathbb R^n\\). I've already demonstrated the utility of this kernel in the previous section while computing dual PCA. In the toy example in Figure 2, the original feature space is \\(\mathcal X = \mathbb R^2\\), the _\\(\mathbb R\\)-Hilbert space_ is \\(\mathcal H = \mathbb R^3\\), and the feature map \\(\phi: \mathcal X \rightarrow \mathcal H\\) is given by
\\[
\phi(x, y) = (x^2, y^2 \sqrt 2 xy).
\\]
You can verify that this feature map corresponds to the so-called _polynomial kernel_
\\[
K(\mathbf x, \mathbf y) = (\langle\mathbf x, \mathbf y\rangle + c)^d
\\]
when \\(c = 0\\) and \\(d = 2\\).


The most commonly used non-linear kernel is the so-called _radial basis function kernel_ \\(K: \mathbb R^n \times \mathbb R^n \rightarrow \mathbb R\\),
\\[
K(\mathbf x, \mathbf y) = \exp\left(\gamma\lVert \mathbf x - \mathbf y \rVert_2^2\right),
\\]
where \\(\gamma > 0\\) is a parameter. As in the case of the quadratic polynomial kernel, you can explicitly write down the corresponding feature map \\(\phi\\), as described in this Stack Exchange post. A less common example is the so-called _Laplacian kernel_
\\[
K(\mathbf x, \mathbf y) = \exp(\gamma \lVert \mathbf x - \mathbf y \rVert_1).
\\]
Intuitively, both of these kernels provide a measure of similarity between two data points. If \\(\mathbf x\\) and \\(\mathbf y\\) are close together, then the distance between these points will be small hence the value of \\(K(\mathbf x, \mathbf y)\\) will be close to 1. Conversely, if \\(\mathbf x\\) and \\(\mathbf y\\) are far apart, then the distance between them will be large and the value of \\(K(\mathbf x, \mathbf y)\\) will be close to zero.


In the current context of genomic data, I'll use the radial basis kernel function

 This makes sense given our original definition of a kernel. By definition,











In the present scenario, not just any feature map \\(\phi\\) will suffice. We have to add the constraint that \\(\phi\\) maps into an inner product space, that t

a continuous and symmetric map \\(K: \mathcal X \times \mathcal X \rightarrow \mathbb R\\) satisfying the property that
\\[
\langle\phi(\mathbf x), \phi(\mathbf y)\rangle.
\\]


A result from functional analysis known as _Mercer's condition_ then guarantees that
 Suppose \\(K: \mathcal X \times \mathcal X \rightarrow \mathbb R\\) is a

I've used scare quotes here because

 Still, we can understand kernel PCA by way of comparison with dual PCA


 \\(\Phi(X) \in \mathbb R^{m \times d}\\) (I use scare quotes since \\(d\\) is possibly infinite).


<!-- Visualization -->
<h3><a id="visualization">Visualization</a></h3>

Due to the large number of populations included in the study, it's difficult to get a feel for the data using standard Python plotting libraries such as matplotlib. I thus created a Dash app to display the interactive Plotly graph displayed below. Dropdown menus allow the user to select the type of PCA (dual or kernel), whether to include or exclude the X and Y chromosomes, and which set of classes to group the data by (population, super population, or gender). You can click and drag to rotate the plot in three dimensions and zoom in and out using your trackpad's scrolling feature. You can also click on each class in the legend on the right to display or hide that class in the graph. Double-clicking will display only points from a particular class and hide points from all other classes. To get more information about a particular data point, simply hover the cursor above the point to view its coordinates and class.

<div class="iframe-container">
  <iframe src="https://visualize-one-k-genomes-3d.herokuapp.com/"></iframe>
</div>

The above plot provides several fascinating insights into the data. First, I was surprised to discover that the plots for dual and kernel PCA look more or less the same. This is not in general the case, and in the current context, I suspect that the similarity is a result of the high dimension of the feature space and the sparse nature of the data. Second, when both sex chromosomes are included, the first principal component of the data is clearly interpretable as a gender axis. Male and female genomes appear to belong to distinct planes clearly separated along the first principal component, with a "V"-like structure mirrored in both planes.

When the sex chromosomes are excluded, each of the five super populations appears to have its own locus in the graph, although some super populations are more disperse than others (in particular, the class of Admixed Americans is quite spread out, an observation that accords with the diversity of heritages found in this super population). Moreover, the African, European, and Admixed American super populations (which historically have had close interactions through colonization and the slave trade) form a plane, with populations of mixed heritage appearing in intermediate locations. For example, the African-American and Afro-Caribbean populations lie mostly on the axis between the native European and native African populations, while Colombians occupy an intermediate position between all three super population loci. In contrast, the East and South Asian super populations do not have the same history of interaction with other groups and appear as relatively isolated clusters. Lastly, the data appears to resolve even geographically proximate populations into distinct clusters (for example, West African populations can be separated into nearby yet discernible clusters).



<div class="iframe-container">
  <iframe src="https://visualize-one-k-genomes-2d.herokuapp.com/"></iframe>
</div>

More observations.


## Remarks

A Hilbert space \\(\mathcal H\\) is a real or complex inner product space, and the that is also a complete metric space

In the current case, the number of samples \\(m = 2504\\) was small enough so that the Gram matrix easily fit in memory and its eigendecomposition could be quickly computed. If we had more samples, we could have resorted to an approximative technique such as the Nystr&ouml;m extension to compute the top eigenvectors without explicitly forming the Gram matrix.

This motivation is perhaps less compelling in the current context, since our original feature space is already high-dimensional. Still, let's suppose for the sake of argument that we wish to map our data into a feature space of (possibly infinite) dimension \\(d\\).

 This clever maneuver is referred to as the "kernel trick," and kernel PCA is simply a generalization of dual PCA that allows the use of non-linear kernels.

## References

Mathematics -- PCA in High Dimensions
https://www.youtube.com/watch?v=NhrhppL4suE


Stack Exchange
https://stats.stackexchange.com/questions/14002/whats-the-difference-between-principal-component-analysis-and-multidimensional/132731#132731
