#pragma once

#include <scope/base/Types.h>
#include <memory>
#include <vector>

namespace scope {
class Link {
 public:
  std::string mName;

  int mId;
  int mParent;
  int mJoint;

  std::vector<int> mvChildren;

 public:
  Link() : mName(""), mId(-1), mParent(-1), mJoint(-1){};

  const std::string& name() const { return mName; };

  const int id() const { return mId; };

  const int parent() const { return mParent; }

  const int joint() const { return mJoint; }

  const std::vector<int>& children() const { return mvChildren; }
};

using LinkPtr = std::shared_ptr<Link>;
using LinkConstPtr = std::shared_ptr<const Link>;
}  // namespace scope
