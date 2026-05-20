// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE Main
#include "inc/Test.h"

#include <boost/test/tree/visitor.hpp>
#include <string>
#ifdef TIKV
#include <absl/synchronization/mutex.h>
#endif

using namespace boost::unit_test;

class SPTAGVisitor : public test_tree_visitor
{
  public:
    void visit(test_case const &test)
    {
        std::string prefix(2, '\t');
        std::cout << prefix << "Case: " << test.p_name << std::endl;
    }

    bool test_suite_start(test_suite const &suite)
    {
        std::string prefix(1, '\t');
        std::cout << prefix << "Suite: " << suite.p_name << std::endl;
        return true;
    }
};

struct GlobalFixture
{
    GlobalFixture()
    {
        // [PERF] Disable absl::Mutex deadlock detector. Default mode (kReport)
        // adds GraphCycles bookkeeping under a global spinlock on every Lock();
        // observed to consume ~12% CPU under high worker-thread parallelism in
        // gRPC client paths (perf-recorded 2026-05-06).
#ifdef TIKV
        absl::SetMutexDeadlockDetectionMode(absl::OnDeadlockCycle::kIgnore);
#endif
        SPTAGVisitor visitor;
        traverse_test_tree(framework::master_test_suite(), visitor, false);
    }
};

BOOST_TEST_GLOBAL_FIXTURE(GlobalFixture);
