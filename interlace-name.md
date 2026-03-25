# Why "interlace"?

In weaving, **interlacing** is when threads cross over and under each other —
warp threads run in one direction, weft threads in the other, and they intersect
at every combination. The structure of the fabric emerges from how these threads
jointly interact; you cannot understand the fabric by examining each set of
threads in isolation.

This is exactly the problem the package solves. **Crossed random effects** arise
when grouping factors are not nested but instead cross — every combination of
factor A and factor B can occur. An observation sits at the intersection of its
group memberships, just like a point in a woven fabric sits where a warp thread
meets a weft thread.

Existing tools estimate each random effect independently, like examining threads
in isolation. `interlace` estimates them **jointly** through a shared sparse
design matrix — capturing the actual fabric of the data.

```python
from interlace import fit

model = fit("y ~ x1 + x2 + (1|group_a) + (1|group_b) + (1|group_c)", data=df)
```
