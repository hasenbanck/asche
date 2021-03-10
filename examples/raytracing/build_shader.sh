#!/usr/bin/env sh
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/postprocess.frag -o shader/postprocess.frag.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/postprocess.vert -o shader/postprocess.vert.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/raytrace.rgen -o shader/raytrace.rgen.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/raytrace.rchit -o shader/raytrace.rchit.spv
glslc --target-env=vulkan1.2 --target-spv=spv1.5 -O shader/raytrace.rmiss -o shader/raytrace.rmiss.spv
