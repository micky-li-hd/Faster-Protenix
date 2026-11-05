"""Create per-launch POSIX shared memory only inside a pre-created private mount namespace."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--expected-parent-mnt',required=True);p.add_argument('--size-bytes',required=True,type=int);p.add_argument('--receipt',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.command and a.command[0]=='--';command=a.command[1:];assert command
namespace=os.readlink('/proc/self/ns/mnt');assert namespace!=a.expected_parent_mnt,'Refuse mount in original namespace'
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    prefix=line.split(' - ')[0].split();assert not any(x.startswith(('shared:','master:')) for x in prefix[6:]),'Mount tree is not private'
assert 0<a.size_bytes<=os.statvfs('/dev/shm').f_blocks*os.statvfs('/dev/shm').f_frsize
before=os.stat('/dev/shm').st_dev
subprocess.run(['mount','-t','tmpfs','-o',f'size={a.size_bytes},mode=1777,nosuid,nodev,noexec','protenix-private-shm','/dev/shm'],check=True)
after=os.stat('/dev/shm').st_dev;assert after!=before
r={'status':'PRIVATE_SHM_MOUNTED','parent_namespace':a.expected_parent_mnt,'namespace':namespace,'original_device':before,'private_device':after,'size_bytes':a.size_bytes,'command':command}
with Path(a.receipt).open('x') as f:json.dump(r,f,indent=2)
os.execvpe(command[0],command,os.environ)
