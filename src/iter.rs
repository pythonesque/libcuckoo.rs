use core::mem;
use core::iter::Step;
use core::ops::Add;

pub struct Range<A> {
    start: A,
    end: A,
}

impl<A> Range<A> {
    pub fn new(start: A, end: A) -> Self {
        Range { start: start, end: end }
    }
}

impl<A: Step> Iterator for Range<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        // FIXME #24660: this may start returning Some after returning
        // None if the + overflows. This is OK per Iterator's
        // definition, but it would be really nice for a core iterator
        // like `x..y` to be as well behaved as
        // possible. Unfortunately, for types like `i32`, LLVM
        // mishandles the version that places the mutation inside the
        // `if`: it seems to optimise the `Option<i32>` in a way that
        // confuses it.
        let mut n = self.start.add_one();
        mem::swap(&mut n, &mut self.start);
        if n < self.end {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match Step::steps_between(&self.start, &self.end.add_one()) {
            Some(hint) => (hint, Some(hint)),
            None => (0, None)
        }
    }
}
