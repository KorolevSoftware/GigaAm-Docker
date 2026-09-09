//go:build linux || darwin

package audio

import "syscall"

func FreeBytes(path string) (uint64, error) {
	var s syscall.Statfs_t
	if e := syscall.Statfs(path, &s); e != nil {
		return 0, e
	}
	return uint64(s.Bavail) * uint64(s.Bsize), nil
}
