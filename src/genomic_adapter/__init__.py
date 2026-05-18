# Genomic Diffusion Adapter (GDA) — joint from-scratch training.
#
# Architecture: full MoPaDi backbone UNet (always receives cond=zeros)
# plus a lightweight adapter UNet that learns the genomic residual Δε.
# Both train together; the backbone never sees patient genomic features,
# so it cannot suppress the adapter. No pretrained checkpoint required.
