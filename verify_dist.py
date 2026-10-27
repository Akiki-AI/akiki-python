"""Check licence metadata, model identity and exact bundled bytes; stdlib only."""
from pathlib import Path
import argparse
import email
import hashlib
import json
import re
import sys
import tarfile
import zipfile

root = Path(__file__).resolve().parent
version = re.search(r'^version = "([^"]+)"$', (root/'pyproject.toml').read_text(), re.M)[1]
if sys.argv[1:] == ['--version']:
    print(version)
    raise SystemExit(0)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dist-dir', type=Path, default=root/'dist')
parser.add_argument('--wheel', type=Path, help='Specific platform wheel to verify against the same source archive')
parser.add_argument('--all-wheels', action='store_true', help='Verify all wheels for this version')
parser.add_argument('--require-native', action='store_true', help='Require at least one wheel with the Rust library')
args = parser.parse_args()
if args.wheel and args.all_wheels:
    parser.error('--wheel and --all-wheels are mutually exclusive')
wheels = (sorted(args.dist_dir.glob(f'akiki-{version}-*.whl')) if args.all_wheels else
          [args.wheel or args.dist_dir/f'akiki-{version}-py3-none-any.whl'])
source = args.dist_dir/f'akiki-{version}.tar.gz'
model_dir = root/'src/akiki/models'
expected_id = 'teacher-only623-v2-s1730'
expected_hash = '0fab56bc4ea6725ec881c35973e04813a4963dc5bbf4bf5ccfe8fd7cf468c6f5'
checks = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(data):
    # Core metadata is UTF-8; the legacy email parser otherwise replaces
    # non-ASCII description text such as the microsecond symbol.
    metadata = email.message_from_string(data.decode('utf-8'))
    require(metadata['Name'] == 'akiki' and metadata['Version'] == version, 'Wrong package name/version')
    require(tuple(map(int, metadata['Metadata-Version'].split('.'))) >= (2, 4), 'License-File requires metadata >=2.4')
    require(metadata['License-Expression'] == 'Apache-2.0', 'Wrong licence expression')
    require(metadata.get_all('License-File') == ['LICENSE'], 'Wrong licence files')
    require(metadata['License'] is None, 'Legacy licence metadata present')
    require(metadata['Description-Content-Type'] == 'text/markdown', 'Description must be Markdown')
    require(metadata.get_payload().strip() == (root/'README.md').read_text().strip(),
            'Stale public description: rebuild this artifact with the current README')
    require(set(metadata.get_all('Provides-Extra') or []) == {'onnx', 'pytorch'}, 'Missing engine extras')
    require(set(metadata.get_all('Requires-Dist') or []) == {
        'numpy>=1.24', 'soundfile>=0.12',
        'onnxruntime>=1.17; extra == "onnx"', 'torch>=2.1; extra == "pytorch"',
    }, 'Runtime dependencies must be optional and independently selectable')
    return {k: metadata[k] for k in ['Metadata-Version', 'License-Expression']}


def check_models(read, names):
    prefix = 'akiki/models/'
    local_files = sorted(path for path in model_dir.rglob('*') if path.is_file())
    require({name for name in names if name.startswith(prefix) and not name.endswith('/')} == {prefix+path.relative_to(model_dir).as_posix() for path in local_files}, 'Unexpected/missing bundled model files')
    for path in local_files:
        require(read(prefix+path.relative_to(model_dir).as_posix()) == path.read_bytes(), f'Bundled bytes differ: {path.name}')
    metadata = json.loads(read(prefix+'akiki-vad.json'))
    release = json.loads(read(prefix+'release.json'))
    require(hashlib.sha256(read(prefix+'akiki-vad.onnx')).hexdigest() == expected_hash, 'Wrong released ONNX weights')
    require(metadata['onnx_sha256'] == expected_hash, 'Wrong sidecar checksum')
    require(metadata['model_id'] == release['model_id'] == expected_id, 'Wrong model identity')
    require(metadata['origin']['seed'] == release['selection']['seed'] == 1730, 'Wrong release seed')
    require(metadata['release_selection']['checkpoint_window'] == release['selection']['checkpoint_window'] == 31, 'Wrong release checkpoint')
    require(release['package_version'] == version, 'Release/package version mismatch')
    require(hashlib.sha256(read(prefix+'akiki-vad.pt')).hexdigest() == release['files']['akiki-vad.pt']['sha256'], 'Wrong released PyTorch weights')
    receipt = json.loads(read(prefix+'akiki-vad-native/export.json'))
    for filename, key in [('config.json', 'config_sha256'), ('model.safetensors', 'model_safetensors_sha256')]:
        require(hashlib.sha256(read(prefix+'akiki-vad-native/'+filename)).hexdigest() == receipt[key], 'Wrong Rust model bytes')
    require(receipt['source_checkpoint_sha256'] == release['files']['akiki-vad.pt']['sha256'], 'Rust export belongs to another checkpoint')
    return dict(model_id=expected_id, seed=1730, checkpoint_window=31, onnx_sha256=expected_hash)


require(wheels, 'No wheels found for this version')
require(not re.search(r'/home/|/Users/|/mnt/|[A-Za-z]:\\Users\\', (root/'README.md').read_text()),
        'Public README contains a private machine path')
native_count = 0
checks['wheels'] = {}
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        info = check_metadata(archive.read(f'akiki-{version}.dist-info/METADATA'))
        require(archive.read(f'akiki-{version}.dist-info/licenses/LICENSE') == (root/'LICENSE').read_bytes(), 'Wheel licence differs')
        info['model'] = check_models(archive.read, archive.namelist())
        for path in (root/'src/akiki').glob('*.py'):
            require(archive.read('akiki/'+path.name) == path.read_bytes(), f'Stale Python module: {path.name}')
        libraries = [name for name in archive.namelist() if name.startswith('akiki/native/') and name.endswith(('.so', '.dylib', '.dll'))]
        wheel_metadata = email.message_from_bytes(archive.read(f'akiki-{version}.dist-info/WHEEL'))
        tags = wheel_metadata.get_all('Tag') or []
        python_tag, abi_tag, platform_tag = wheel.stem.rsplit('-', 3)[1:]
        require(set(tags) == {f'{python_tag}-{abi_tag}-{platform}' for platform in platform_tag.split('.')},
                'Wheel filename and internal tags differ')
        if libraries:
            require(wheel_metadata['Root-Is-Purelib'] == 'false', 'Native wheel marked pure')
            require(all(tag.startswith('py3-none-') for tag in tags), 'Wrong native Python/ABI tag')
            for platform in platform_tag.split('.'):
                require(platform.startswith(('manylinux', 'musllinux', 'macosx_', 'win')), 'Native wheel needs a PyPI-compatible platform tag')
                expected = sorted('akiki/native/'+(f'{stem}.dll' if platform.startswith('win') else
                                                  f'lib{stem}.dylib' if platform.startswith('macosx_') else f'lib{stem}.so')
                                  for stem in ('akiki_vad_engine', 'akiki_minilm_engine'))
                require(sorted(libraries) == expected, 'Native wheel needs the VAD and MiniLM libraries for its platform')
            native_count += 1
            info['native_libraries_sha256'] = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(libraries)}
        else:
            require(tags == ['py3-none-any'] and wheel_metadata['Root-Is-Purelib'] == 'true', 'Non-native wheel must be portable')
        info['tags'] = tags
        checks['wheels'][wheel.name] = info
require(not args.require_native or native_count > 0,
        'No native wheel found; build and repair the Rust wheel before publishing')
with tarfile.open(source) as archive:
    def source_read(name):
        return archive.extractfile(f'akiki-{version}/src/'+name).read()
    checks['sdist'] = check_metadata(archive.extractfile(f'akiki-{version}/PKG-INFO').read())
    require(archive.extractfile(f'akiki-{version}/README.md').read() == (root/'README.md').read_bytes(), 'Source README differs')
    require(archive.extractfile(f'akiki-{version}/LICENSE').read() == (root/'LICENSE').read_bytes(), 'Source licence differs')
    source_prefix = f'akiki-{version}/src/'
    checks['sdist']['model'] = check_models(source_read, [name[len(source_prefix):] for name in archive.getnames() if name.startswith(source_prefix) and archive.getmember(name).isfile()])
result_path = root/'verification'/version/('result.json' if args.wheel is None else f'{args.wheel.stem}.json')
result = {}
result.update(version=version, model_id=expected_id, metadata=checks, bundled_model_sha256=expected_hash,
              artifacts={path.name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in [*wheels, source]})
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, indent=2)+'\n')
print(f'Verified akiki {version}: {len(wheels)} wheel(s), {native_count} native, and source archive; current public description and exact seed 1730/window 31 weights.')
