#!/usr/bin/env sh
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/trace.rchit -o shader/trace.rchit.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/trace.rgen -o shader/trace.rgen.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/trace.rmiss -o shader/trace.rmiss.spv
