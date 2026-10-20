# Theory: identification, impossibility, and bounds for thresholded feature multiplicity

> **Status.** Proposition 1, Lemma 2 and Proposition 2 below come with complete
> elementary proofs and are machine-checked numerically in `tests/`
> (`test_schur_horn.py`, `test_jointdiag.py`, `test_metrics_stats_cli.py::test_two_explanations_*`).
> The coverage statements for the *procedures* (§5–§6) are asymptotic arguments
> under stated conditions, **not** uniform-validity theorems. They have not been
> reviewed by a statistician. The plan (p.8) and syllabus (S23) both flag that step
> as the highest-risk part of the project; treat it accordingly.

Notation: `d` neurons, `K` latent factors, `E` environments. For a matrix `A`, `a_k`
is column `k` and `A_G` the columns indexed by `G`. `vech` stacks the upper triangle.

---

## 1. Model and estimand

For environment `e`:

$$h^{(e)} = A z^{(e)} + \varepsilon^{(e)}, \qquad \operatorname{Cov}(z^{(e)}) = \operatorname{diag}(v_e), \qquad \operatorname{Cov}(\varepsilon^{(e)}) = \Sigma_\varepsilon \ \text{(same for all } e).$$

Columns of `A` have unit norm; explanations are equivalent up to column sign and permutation.
The **centred profile** of factor `k` is $c_{k,e} = v_{e,k} - \bar v_k$, and the centred covariances are

$$D_e := C_e - \tfrac1E\textstyle\sum_{e'} C_{e'} = A\,\operatorname{diag}(c_{\cdot,e})\,A^\top \qquad (\text{stationary noise cancels}).$$

**Visibility.** Factor `k` belongs to the target iff $\mathrm{vis}_k = (\tfrac1E\sum_e c_{k,e}^2)^{1/2} \ge v_{\min}$.

**Estimand.** $P_j(\tau) = \#\{k \text{ visible} : |A_{jk}| \ge \tau\}$.

The information set used throughout is the stack $\{D_e\}$, i.e. *second moments*. Every
"identified set" below is relative to that information set and the declared model. Higher
moments (as used by FastICA) can carry more information. They are an explicit extension, not
an implicit assumption.

---

## 2. Covariance-difference identification (established; coroICA, syllabus S11)

If `A` is square and invertible and all visible profiles are pairwise **non-collinear**, the
joint diagonaliser of $\{D_e\}$ recovers `A` up to scale and permutation (Pfister et al.,
JMLR 2019). We do not claim this. The rest of this document concerns what happens when the
condition fails, and what that does to $P_j$.

---

## 3. Proposition 1 — the exact second-moment identified set

**Setting.** `A` square and invertible; every factor visible (a nonzero profile). Partition
the factors into classes of mutually collinear profiles: within a class `G`,
$c_k = \lambda_k u_G$ with a common unit vector $u_G \in \mathbb R^E$, and profiles from
different classes are non-collinear.

**Proposition 1.** A pair $(A', c')$ with $A'$ invertible reproduces $\{D_e\}$ exactly if and
only if, up to permutation of columns,

1. for every singleton class $\{k\}$: $a'_k = \pm a_k$ (after normalisation) and $c'_k = c_k$
   (after the matching rescaling);
2. for every class `G` with $|G| = g \ge 2$:
   $A'_G \operatorname{diag}(\lambda') A_G'^\top = A_G \operatorname{diag}(\lambda) A_G^\top =: \Gamma_G$
   and $c'_k = \lambda'_k u_G$ for $k \in G$.

When all $\lambda_k > 0$, the normalised admissible columns of the class are exactly

$$\mathcal A_G = \{\operatorname{normalize}(M_G R) : R \in O(g)\}, \qquad M_G = A_G \operatorname{diag}(\lambda)^{1/2}.$$

*Proof.* Pick generic weights $w, w'$ and form $S = \sum_e w_e D_e = A \operatorname{diag}(c^\top w) A^\top$
and $S' = \sum_e w'_e D_e$. Because every profile is nonzero, a generic $w'$ makes $S'$
invertible. Then

$$S S'^{-1} = A \operatorname{diag}(r) A^{-1}, \qquad r_k = \frac{c_k^\top w}{c_k^\top w'} .$$

So the columns of `A` are eigenvectors of $SS'^{-1}$, a matrix fixed by the data. For generic
weights, $r_k = r_l$ iff $c_k \parallel c_l$. Hence the eigenvalue of a singleton class is
simple, and any alternative $A'$ must have $a'_k \propto a_k$. A class `G` has an eigenvalue
of multiplicity `g`, and $A'_G$ must span the same eigenspace: $A'_G = A_G T$ with `T`
invertible. Matching the class contribution in every environment gives
$T \operatorname{diag}(c'_{G,e}) T^\top = u_{G,e}\Lambda$ for all `e`, with
$\Lambda = \operatorname{diag}(\lambda)$. Taking an `e` with $u_{G,e} \neq 0$ shows
$T^{-1}\Lambda T^{-\top}$ is diagonal, hence $c'_{G,e} = u_{G,e}\lambda'$ and
$T\operatorname{diag}(\lambda')T^\top = \Lambda$. If $\Lambda \succ 0$, Sylvester's law of
inertia forces $\lambda' \succ 0$, and every solution has the form
$T = \Lambda^{1/2} R\, \operatorname{diag}(\lambda')^{-1/2}$ with `R` orthogonal. The column
scaling disappears on normalisation. The converse is direct substitution. ∎

*Remarks.* (i) If the $\lambda_k$ in a class have mixed signs, `R` ranges over the
indefinite orthogonal group $O(p,q)$. That group is non-compact, so normalised columns can
approach each other. `polyid` then uses the span outer bound (§4, last paragraph).
(ii) Invisible factors (zero profiles) are absent from $\{D_e\}$ and are not part of the
target. The visible columns remain identified: if $\tilde B$ jointly diagonalises the stack,
its rows for invisible components annihilate every visible column, so the visible columns of
$\tilde B^{-1}$ are the visible columns of `A` up to scale (whatever the invisible rows are).
(iii) E0 constructs two explicit members of $\mathcal A_G$ whose full population covariances
agree to $5.6\times10^{-17}$ but whose $P_j$ differ (`results/e0_theory_gate/e0_summary.json`).

---

## 4. Lemma 2 — exact multiplicity range of an unidentified class (Schur–Horn)

Column `i` of $M_G R$ reaches the threshold on neuron `j` iff
$(m_j^\top r_i)^2 \ge \tau^2\, r_i^\top M_G^\top M_G r_i$, i.e.
$r_i^\top Q_j r_i \ge 0$ with $Q_j = m_j m_j^\top - \tau^2 M_G^\top M_G$ ($m_j^\top$ = row `j`
of $M_G$). So the class contributes $N_j(R) = \#\{i : (R^\top Q_j R)_{ii} \ge 0\}$.

By the **Schur–Horn theorem** (Schur 1923; Horn 1954, real symmetric / orthogonal case),
$\{\operatorname{diag}(R^\top Q R) : R \in O(g)\}$ is exactly the set of vectors majorised by
the eigenvalue vector $\lambda(Q)$. Write $s = \operatorname{tr} Q$.

**Lemma 2.**
* $\max_R N_j = g$ if $s \ge 0$. Otherwise it is the largest `m` for which
  $x^{(m)} = (0,\dots,0,\ \tfrac{s}{g-m},\dots,\tfrac{s}{g-m})$ (`m` zeros) satisfies
  $x^{(m)} \prec \lambda(Q)$.
* $\min_R N_j = 0$ if $s < 0$. Otherwise it is $g-n$ for the largest `n` such that
  $(-\epsilon,\dots,-\epsilon,\ \tfrac{s+n\epsilon}{g-n},\dots) \prec \lambda(Q)$ for some small
  $\epsilon>0$.

*Proof (max; the min is symmetric).* If $s \ge 0$, the barycentre $(s/g,\dots,s/g)$ is
majorised by every vector with sum `s`, so it is attainable and all entries are $\ge 0$. If
$s<0$, let `x` be any attainable diagonal with at least `m` non-negative entries, sorted in
decreasing order. For $k \le m$ the partial sums of `x` are $\ge 0$, which equals the partial
sums of $x^{(m)}$. For $k>m$, the average of the smallest $g-k$ entries is at most the
average of the smallest $g-m$ entries, and the latter sum to at most `s`. So
$\sum_{i>k}x_i \le \tfrac{g-k}{g-m}s$, hence $\sum_{i\le k} x_i \ge \tfrac{k-m}{g-m}s$, the
partial sum of $x^{(m)}$. So $x^{(m)} \prec x \prec \lambda(Q)$, and by transitivity
$x^{(m)}$ is attainable. It has `m` entries equal to $0 \ge 0$. ∎

Because the classes rotate independently, the **identified set of $P_j$** is
$[\sum_{\text{singletons}} \mathbb 1\{|a_{jk}|\ge\tau\} + \sum_G \min_R N_j,\ \ \cdots + \sum_G \max_R N_j]$
(interval hull). `bounds/schur_horn.py` implements this. Exactness for `g = 2` is checked
against a 20 000-angle sweep, and containment for `g = 3` against random rotations
(`tests/test_schur_horn.py`).

**Span outer bound.** If within-class latents may be correlated, or the signs are mixed, any
basis of $\operatorname{span}(A_G)$ is admissible. The range is then $[0, g]$ when
$\|P_{\mathrm{span}} e_j\| > \tau$ and $[0,0]$ otherwise ($g\ge2$). This is valid but wider
than necessary.

---

## 5. Proposition 2 — co-occurrence that no second-moment method can detect

**Setting.** A block `G` of factors shares one activity switch $b \sim \mathrm{Bern}(p_e)$,
with independent magnitudes $s_k$ (means $\mu_k$, variances $\sigma_k^2$, constant across
environments). Then

$$\operatorname{Cov}(z_G^{(e)}) = p_e \operatorname{diag}(\sigma^2) + p_e(1-p_e)\, \mu\mu^\top ,$$

since $\operatorname{Cov}(b s_k, b s_l) = p\,E[s_k s_l] - p^2\mu_k\mu_l$. Every block covariance
therefore has the form $A_G(\alpha_e \Delta + \beta_e\, \mu\mu^\top)A_G^\top$ with the fixed
diagonal $\Delta = \operatorname{diag}(\sigma^2) \succ 0$, $\alpha_e = p_e$ and
$\beta_e = p_e(1-p_e)$. The centred versions have the same form.

**Proposition 2.** For any block size $g \ge 2$, there is an *uncorrelated-factor* (hence
second-moment-indistinguishable from independent) explanation
$(A'_G, c'_G)$ of the same block, with $\operatorname{span}(A'_G) = \operatorname{span}(A_G)$,
that reproduces every $D_e$ exactly. Generically $A'_G \neq A_G$ up to sign and permutation,
and its $P_j$ differ.

*Proof.* $P = A_G \Delta A_G^\top$ is positive definite on $\operatorname{span}(A_G)$, and
$S = (A_G\mu)(A_G\mu)^\top$ is symmetric. Two such forms are simultaneously diagonalisable:
there is `B` with $BPB^\top$ and $BSB^\top$ both diagonal. Then every block term
$\alpha_e P + \beta_e S$ is diagonalised by `B`. Setting $A'_G = B^{-1}$ (restricted to the
span) and latent variances $v'_{k,e} = \alpha_e (BPB^\top)_{kk} + \beta_e (BSB^\top)_{kk}$ —
non-negative, because both diagonals are those of PSD matrices and $\alpha_e,\beta_e \ge 0$ —
gives a model with uncorrelated latents and the same covariances in every environment, hence
the same $D_e$. ∎

**Corollary (exchangeable dependence).** The proof uses only that every block covariance lies in
the span of two fixed matrices, one of them positive definite. That holds for *any* exchangeable
dependence within the block. If the members share activity rates and magnitude laws, then
$\operatorname{Cov}(z_G^{(e)}) = a_e I + b_e \mathbf 1\mathbf 1^\top$ for any switch copula. Partial
co-occurrence with shared rates is therefore also second-moment-undetectable. The E2 pilot's
`cooccurrence_partial` condition (copula 0.5, shared rates) confirms it: the infinite-sample
procedure fits exactly and misses the truth.

**Consequences.** Perfect co-occurrence is not "detectably misspecified": it is
*observationally equivalent* to a clean independent model with different loadings. A
second-moment procedure can return a narrow, confidently wrong answer here. This holds for
ISMB, the AR-grid set and every joint-diagonalisation method, at any sample size.
`identification.py` labels the case `misspecified` with `misfit_detectable = False`, and E2/E4
report how each method behaves on it. Partial co-occurrence (copula correlation < 1) makes
$E[b_k b_l]$ a nonlinear function of $p_e$. It is then generally detectable once enough
environments are observed, which E2 measures. Repairs need information outside the declared
set: higher-order cumulants, interventions, or design knowledge that the block is not
co-activated.

---

## 6. Remark 3 — the pairwise rank test is first-order robust to estimating the subspace

For components `k, l` let $M_e = Q D_e Q^\top$ with `Q` a basis of the estimated pair
subspace. Under $H_0$ (collinear pair) $M_e = u_e\Gamma$. If $\hat Q = Q + \mathcal E$, then

$$\hat Q D_e \hat Q^\top - Q D_e Q^\top = \mathcal E D_e Q^\top + Q D_e \mathcal E^\top + O(\|\mathcal E\|^2),$$

and $D_e Q^\top = A \operatorname{diag}(c_e) A^\top Q^\top$. Only the pair's own columns survive
($Q a_m = 0$ for $m \notin G$), and those have $c_{m,e} = \lambda_m u_e$. So the first-order
perturbation equals $u_e \times(\text{fixed matrix})$: it lies in the tangent space of the
rank-1 manifold at the truth. The Cragg–Donald statistic is therefore unaffected to first
order, and treating $\hat Q$ as fixed is justified. Empirically, the test rejected a truly
collinear pair 5.5 % of the time at nominal 5 % (200 replicates, estimated basis), with power
0.95 on a separated pair (`tests/test_bounds.py` re-checks a 60-replicate version).

---

## 7. What the procedures' coverage rests on

### ISMB (`bounds/ismb.py`)

$\Pr(L_j \le P_j \le U_j\ \forall j)\to \ge 1-\alpha$ under the following conditions.
(a) The declared model holds outside merged groups. (b) Every truly collinear pair is merged:
the pair tests are Bonferroni-controlled at $\alpha_{\text{pair}}$. (c) Separated factors are
**strongly** identified, so the bootstrap band for their loadings is valid. (d) Bootstrap
quantiles cover the group ranges. (e) The visibility bands are valid: they are analytic and
first-order exact for invisible directions.

Condition (c) fails *non-uniformly*. Near-collinear but distinct profiles are exactly the
regime where Stoye (2009, S21) shows Imbens–Manski-type intervals can undercover. GMS
selection (Andrews & Soares 2010) is pointwise valid but is a further non-uniform step. E2
(`weak_diversity`, `near_duplicate_columns`) and E4 (`weak_diversity`) measure this regime
rather than assume it away.

### AR-grid (`bounds/ar_grid.py`, `d = 3`)

At the true `A`, the profiled efficient criterion $J(A)$ is asymptotically
$\chi^2_{\dim m - qK}$, regardless of identification strength, because the nuisance enters
linearly and the weighting is efficient. So $C_n = \{A \in \mathcal G: J(A) \le
\chi^2_{1-\alpha}\}$ covers the truth with asymptotic probability $\ge 1-\alpha$ **when the
truth is on the grid**. Projection onto $P_j$ is conservative. An empty $C_n$ is evidence
against the model. Off-grid explanations are invisible to it, so the set can be narrower than
the identified set. The statement is grid-restricted, as the plan requires.

### What is *not* established

* Uniform (over data-generating processes) validity of ISMB near the collinearity boundary.
* Any guarantee in the overcomplete regime ($K > d$). SOBIUM / CP-uniqueness conditions
  (X9) would be needed.
* Inference for the discontinuous $P_j$ beyond projection. Calibrated projection
  (Kaido–Molinari–Stoye 2019, S23) covers smooth functions. `estimand.smooth_multiplicity`
  is the surrogate on which it would apply; linking it back to $P_j$ is open.
* Anything about semantic concepts. $P_j$ is statistical loading multiplicity under a
  declared model.

## 8. Cumulant-aware ISMB (`ismb_cum`): a different information set

Second moments cannot separate two factors whose variance profiles are collinear
(Proposition 1). Fourth-order cumulants can. Under the declared model with independent latents
$z_k$ and Gaussian noise, every per-environment slice of the fourth-order cumulant tensor of
$h$ is

$$\mathcal C_4^{(e)}(e_p e_p^\top) = A\,\operatorname{diag}\big(\kappa_k^{(e)} a_{pk}^2\big)A^\top,$$

so it is diagonalised by the same unmixing matrix as the centred covariances. Gaussian noise
contributes nothing, whether stationary or not. Within a group of factors whose second-moment
profiles are collinear, the rotation left free by Proposition 1 is pinned down unless **two or
more** members of the group are Gaussian in every environment. This is the identifiability
condition of ICA (Comon 1994): at most one Gaussian source per unidentified block.

`ismb_cum` changes ISMB in four places. Everything else (visibility bands, loading bands,
Schur–Horn group ranges, misfit gate) is unchanged.

1. **Basis.** The covariance differences and the $E \cdot d$ cumulant slices are diagonalised
   jointly. Each matrix is weighted by the inverse of its estimated sampling-noise scale
   (precision weighting). Unit-norm weighting, the first choice, gave pure-noise slices full
   weight when latents are near-Gaussian, and widened intervals to near-trivial (lab notebook §12).
2. **Merge rule.** A pair with collinear second-moment profiles is merged only if a test of
   "excess fourth cumulant = 0 in every environment" fails to reject for **both** members. The
   test sums squared delta-method $z$-scores over environments and is referred to $\chi^2_E$.
   Its level is `share_gauss`, Bonferroni over components, taken out of ISMB's pair-test share.
   A wrong decision to trust identification therefore costs at most that share of $\alpha$.
3. **Dependence.** For every pair, a cross-cumulant test of $\operatorname{cum}(u_k,u_k,u_l,u_l)=0$
   in every environment. It is insensitive to first-order unmixing error, because an error
   $\varepsilon$ enters at order $\varepsilon^2$. Pairs it flags are merged under the span
   (correlated) outer bound, which only widens. This is how the co-occurrence of
   **Proposition 2**, invisible to every second-moment method, becomes detectable: a shared
   switch makes $\operatorname{cum}(z_k,z_k,z_l,z_l) \neq 0$.
4. **Second-moment tests on a second-moment basis.** The pairwise collinearity, dependence and
   misfit tests were derived for a basis fitted to the covariance stack. The combined fit is a
   compromise, and its second-moment residuals made the misfit gate fire on clean data. The
   tests are therefore run on a second-moment refinement warm-started from the combined
   solution.

**What its coverage rests on.** Conditions (a)–(e) of §7, with (b) replaced by: every truly
unidentified pair is merged, unless the Gaussianity test wrongly rejects (level `share_gauss`).
(c) now reads: factors separated *by cumulants* are strongly identified. That fails
non-uniformly as latents approach Gaussianity, the fourth-order analogue of the Stoye regime.
E11 measures it with `latent_gaussianity` ∈ {0.5, 0.9, 1.0}. A Gaussian or heavy-tailed
**noise** misspecification (e.g. Student-t noise) contaminates the cumulant slices. The
procedure then widens through merges or the misfit gate, but nothing guarantees it does. E11
reports `noise_student_t`.

**Status: exploratory.** It uses a higher-moment information set (`Access.higher_moments`). It
is a different estimator, compared side by side with ISMB, and the statistician review that
gates ISMB's claims (preregistration §3) applies to it too.
