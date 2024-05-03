# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script for installation and distribution.
You can use environment variable `SPTAG_RELEASE` to set release version.
If release version is not set, default to a development build whose version string will be `0.0.0.dev`.
## Prepare Environment ##
Install dependencies:
  $ pip install -U -r setup.txt
## Development ##
Build and install for development:
  $ python setup.py develop
Uninstall:
  $ pip uninstall sptag
Remove generated files: (use "--all" to remove toolchain and built wheel)
  $ python setup.py clean [--all]
## Release ##
Build wheel package:
  $ SPTAG_RELEASE=1.0 python setup.py bdist_wheel -p win_amd64
Where "1.0" is version string and "win_amd64" is platform.
The platform may also be "manylinux1_x86_64".
"""

from distutils.cmd import Command
from distutils.command.build import build
from distutils.command.clean import clean
import glob
import os
import shutil
import sys

import setuptools
from setuptools.command.develop import develop

release = os.environ.get('SPTAG_RELEASE')
nuget_release = os.environ.get('NUGET_RELEASE')
python_version = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
print ("Python version:%s" % python_version)

if 'bdist_wheel' in sys.argv:
  if not any(arg.startswith('--python-tag') for arg in sys.argv):
      sys.argv.extend(['--python-tag', 'py%d%d'%(sys.version_info.major, sys.version_info.minor)])
print (sys.argv)

def _setup():
    setuptools.setup(
        name = 'sptag',
        version = release or '0.0.0.dev',
        description = 'SPTAG: A library for fast approximate nearest neighbor search',
        long_description = open('README.md', encoding='utf-8').read(),
        long_description_content_type = 'text/markdown',
        url = 'https://github.com/Microsoft/SPTAG',
        author = 'Microsoft SPTAG Team',
        author_email = 'cheqi@microsoft.com',
        license = 'MIT',
        include_package_data=True,
        classifiers = [
            'License :: OSI Approved :: MIT License',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research',
        ],

        packages = _find_python_packages(),
        python_requires = '>=3.6',
        install_requires = ['numpy'],

        cmdclass = {
            'build': Build,
            'clean': Clean,
            'develop': Develop,
        }
    )

def _find_python_packages():
    if os.path.exists('sptag'): shutil.rmtree('sptag')

    if os.path.exists('Release'):
        shutil.copytree('Release', 'sptag')
    elif os.path.exists(os.path.join('x64', 'Release')):
        shutil.copytree(os.path.join('x64', 'Release'), 'sptag')

        if not os.path.exists('lib'): os.mkdir('lib')
        if not os.path.exists('lib\\net472'): os.mkdir('lib\\net472')
        for file in glob.glob(r'x64\\Release\\Microsoft.ANN.SPTAGManaged.*'):
            print (file)
            shutil.copy(file, "lib\\net472\\")
        sfiles = ''
        for framework in ['net5.0', 'net472']:
            if os.path.exists("lib\\%s\\Microsoft.ANN.SPTAGManaged.dll" % framework):
                sfiles += '<file src="lib\\%s\\Microsoft.ANN.SPTAGManaged.dll" target="lib\\%s\\Microsoft.ANN.SPTAGManaged.dll" />' % (framework, framework)
                sfiles += '<file src="lib\\%s\\Microsoft.ANN.SPTAGManaged.pdb" target="lib\\%s\\Microsoft.ANN.SPTAGManaged.pdb" />' % (framework, framework)
        if os.path.exists('x64\\Release\\libzstd.dll'):
            sfiles += '<file src="x64\\Release\\libzstd.dll" target="runtimes\\win-x64\\native\\libzstd.dll" />'
        if os.path.exists('x64\\Release\\Ijwhost.dll'):
            sfiles += '<file src="x64\\Release\\Ijwhost.dll" target="runtimes\\win-x64\\native\\Ijwhost.dll" />'
        sfiles += '<file src="SPTAG.targets" target="build\\MSSPTAG.Managed.Library.targets" />'
        f = open('sptag.nuspec', 'w')
        spec = '''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2011/08/nuspec.xsd">
  <metadata>
    <id>MSSPTAG.Managed.Library</id>
    <version>%s</version>
    <title>MSSPTAG.Managed.Library</title>
    <authors>cheqi,haidwa,mingqli</authors>
    <owners>cheqi,haidwa,mingqli</owners>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <licenseUrl>https://github.com/microsoft/SPTAG</licenseUrl>
    <projectUrl>https://github.com/microsoft/SPTAG</projectUrl>
    <description>SPTAG (Space Partition Tree And Graph) is a library for large scale vector approximate nearest neighbor search scenario released by Microsoft Research (MSR) and Microsoft Bing.</description>
    <copyright>Copyright @ Microsoft</copyright>
  </metadata>
  <files>
%s
  </files>
</package>
''' % (nuget_release, sfiles)
        f.write(spec)
        f.close()

        fwinrt = open('sptag.winrt.nuspec', 'w')
        spec = '''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2011/08/nuspec.xsd">
  <metadata>
    <id>MSSPTAG.WinRT</id>
    <version>%s</version>
    <title>MSSPTAG.WinRT</title>
    <authors>cheqi,haidwa,mingqli</authors>
    <owners>cheqi,haidwa,mingqli</owners>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <licenseUrl>https://github.com/microsoft/SPTAG</licenseUrl>
    <projectUrl>https://github.com/microsoft/SPTAG</projectUrl>
    <description>SPTAG (Space Partition Tree And Graph) is a library for large scale vector approximate nearest neighbor search scenario released by Microsoft Research (MSR) and Microsoft Bing.</description>
    <copyright>Copyright @ Microsoft</copyright>
    <dependencies>
      <group targetFramework="uap10.0">
        <dependency id="Zstandard.dyn.x64" version="1.4.0" />
      </group>
      <group targetFramework="native">
        <dependency id="Zstandard.dyn.x64" version="1.4.0" />
      </group>
    </dependencies>
  </metadata>
  <files>
    <file src="Wrappers\\WinRT\\SPTAG.WinRT.targets" target="build\\native" />
    <file src="x64\Release\\SPTAG.WinRT\\SPTAG.winmd" target="lib\\uap10.0" />
    <file src="x64\Release\\SPTAG.WinRT\\SPTAG.dll" target="runtimes\\win10-x64\\native" />
    <file src="readme.md" />
    <file src="LICENSE" />
  </files>
</package>
''' % (nuget_release)
        fwinrt.write(spec)
        fwinrt.close()

    f = open(os.path.join('sptag', '__init__.py'), 'w')
    f.close()
    return ['sptag']

class Build(build):
    def run(self):
        if not release:
            sys.exit('Please set environment variable "SPTAG_RELEASE=<release_version>"')

        open('sptag/version.py', 'w').write(f"__version__ = '{release}'")
        super().run()

class Develop(develop):
    def run(self):
        open('sptag/version.py', 'w').write("__version__ = '0.0.0.dev'")
        super().run()

class Clean(clean):
    def finalize_options(self):
        self._all = self.all
        self.all = True  # always use `clean --all`
        super().finalize_options()

    def run(self):
        super().run()
        shutil.rmtree('sptag.egg-info', ignore_errors=True)
        if self._all:
            shutil.rmtree('dist', ignore_errors=True)

if __name__ == '__main__':
    _setup()
