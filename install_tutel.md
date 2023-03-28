# Install Tutel

Our method depends on a very early [vesion](https://github.com/microsoft/tutel/tree/56dbd664341cf6485c9fa292955f77d3ac918a65) of Tutel. The commit id is 56dbd664341cf6485c9fa292955f77d3ac918a65. The Tutel has changed a lot since then, so please make sure you download the correct version.

After dwonload the code, run:

```sh
cd tutel
python3 ./setup.py install
```

You may need to change the cuda version. Just search `/usr/local/cuda` in an editor and change all of them to your cuda location such as `/usr/local/cuda-11.1`.

You may need to install NCCL library. Please follow its website.

If you intsall NCCL manually, you should add the include path to the `setup.py` of tutel. You can add `ext_args['cxx'] += ['-I/you/path/to/NCCL/include']` after Line 68 in the `setup.py`.
If you have several local libraries installed on your own directory and you need them for the compilation, you should add `library_dirs += ['/you/local/lib']` after Line 115 in the `setup.py`.